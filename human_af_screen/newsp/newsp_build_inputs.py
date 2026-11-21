#!/usr/bin/env python3
"""Stage 2 inputs, built ON THE CLUSTER from public data (nothing is copied from the Mac): AlphaFold DB complexes metadata (EBI FTP) filtered to the five taxa, STRING v12 physical links + aliases, SIFTS, UniProt reference proteomes
(REST), and sequences for every candidate protein (proteome FASTA, else AlphaFold DB API). Then the same construction as newsp_tiers.py: per species data/newsp/SP/{tiers.tsv.gz, cand_proteins.fasta, tiers_meta.json}.
Fails fast if the node has no outbound access. cwd = FastPPI2."""
import gzip, json, os, subprocess, sys, time, urllib.request, concurrent.futures as cf, numpy as np, pandas as pd
SP = {"soybean": (3847, "UP000008827"), "arabidopsis": (3702, "UP000006548"), "rat": (10116, "UP000002494"), "rice": (39947, "UP000059680"), "zebrafish": (7955, "UP000000437")}
DL = "data/newsp_build"; os.makedirs(DL, exist_ok=True); t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)
AF = "https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/heterodimer_metadata.csv"
def fetch(url, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return
    for k in range(retries):
        r = subprocess.run(["curl", "-sSL", "--fail", "-m", "3600", "-o", dest + ".part", url])
        if r.returncode == 0: os.replace(dest + ".part", dest); return
        time.sleep(10)
    sys.exit(f"DOWNLOAD FAILED: {url}")
r = subprocess.run(["curl", "-sIL", "-m", "30", "-o", "/dev/null", "-w", "%{http_code}", AF], capture_output=True, text=True)
if r.stdout.strip() != "200": sys.exit(f"NOT REACHABLE from this node (http {r.stdout.strip()}): {AF}")
log("outbound access OK; downloading")
fetch(AF, f"{DL}/heterodimer_metadata.csv"); log(f"AFDB metadata {os.path.getsize(DL + '/heterodimer_metadata.csv'):,} B")
fetch("https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz", f"{DL}/pdb_chain_uniprot.tsv.gz")
for n, (tx, up) in SP.items():
    for kind in ("protein.physical.links.detailed", "protein.aliases"): fetch(f"https://stringdb-downloads.org/download/{kind}.v12.0/{tx}.{kind}.v12.0.txt.gz", f"{DL}/{tx}.{kind}.v12.0.txt.gz")
    fetch(f"https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28proteome%3A{up}%29", f"{DL}/{n}.fasta")
log("downloads done")
cols = ["uniprot_ac_1", "uniprot_ac_2", "tax_id_1", "tax_id_2", "ipTM", "max_ipSAE", "max_pDockQ2_AB", "passes_quality_threshold"]; txs = {v[0] for v in SP.values()}; parts = []
for ch in pd.read_csv(f"{DL}/heterodimer_metadata.csv", usecols=cols, chunksize=500000):
    parts.append(ch[ch.tax_id_1.isin(txs) & (ch.tax_id_1 == ch.tax_id_2)])
T = pd.concat(parts); T["passes"] = T.passes_quality_threshold.astype(bool); chk = ((T.max_ipSAE >= 0.6) & (T.max_pDockQ2_AB >= 0.23)); log(f"filtered rows {len(T):,}; label equals the ipSAE >= 0.6 & pDockQ2 >= 0.23 rule on {int((chk == T.passes).sum()):,} of {len(T):,}")
assert (chk == T.passes).all(), "high-confidence label differs from the rule"
entry_acc = {}
with gzip.open(f"{DL}/pdb_chain_uniprot.tsv.gz", "rt") as f:
    next(f); next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) > 2: entry_acc.setdefault(p[0], set()).add(p[2])
def read(path):
    d, k = {}, None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"): h = l[1:].split()[0]; k = h.split("|")[1] if "|" in h else h; d[k] = ""
        elif l and k: d[k] += l.rstrip("*")
    return d
def get(a):
    for _ in range(3):
        try:
            j = json.loads(urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{a}", timeout=60).read().decode()); j = j[0] if isinstance(j, list) else j; return a, j.get("uniprotSequence", "")
        except Exception: time.sleep(2)
    return a, ""
for n, (tx, up) in SP.items():
    d = T[T.tax_id_1 == tx].copy(); d = d[d.uniprot_ac_1 != d.uniprot_ac_2].drop_duplicates(["uniprot_ac_1", "uniprot_ac_2"]); al = {}
    for l in gzip.open(f"{DL}/{tx}.protein.aliases.v12.0.txt.gz", "rt"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2] == "UniProt_AC": al.setdefault(p[0], set()).add(p[1])
    sc = {}
    for l in gzip.open(f"{DL}/{tx}.protein.physical.links.detailed.v12.0.txt.gz", "rt"):
        p = l.split()
        if p[0] == "protein1": continue
        for a in al.get(p[0], ()):
            for b in al.get(p[1], ()):
                if a != b: k = (a, b) if a < b else (b, a); sc[k] = max(sc.get(k, 0), int(p[-1]))
    key = [(a, b) if a < b else (b, a) for a, b in zip(d.uniprot_ac_1, d.uniprot_ac_2)]; d["string"] = [sc.get(k, np.nan) for k in key]; n_nostring = int(d.string.isna().sum()); d = d[d.string.notna()].copy()
    accs = set(d.uniprot_ac_1) | set(d.uniprot_ac_2); pdbpairs = set()
    for e, s in entry_acc.items():
        s2 = sorted(s & accs)
        for i in range(len(s2)):
            for j in range(i + 1, len(s2)): pdbpairs.add((s2[i], s2[j]))
    d["pdb"] = [k in pdbpairs for k in [(a, b) if a < b else (b, a) for a, b in zip(d.uniprot_ac_1, d.uniprot_ac_2)]]
    seqs = read(f"{DL}/{n}.fasta"); miss = sorted(accs - set(seqs)); got = 0
    if miss:
        with cf.ThreadPoolExecutor(8) as ex:
            for a, s in ex.map(get, miss):
                if s: seqs[a] = s; got += 1
    ok = d.uniprot_ac_1.isin(seqs) & d.uniprot_ac_2.isin(seqs); n_noseq = int((~ok).sum()); d = d[ok]; accs = set(d.uniprot_ac_1) | set(d.uniprot_ac_2)
    o = f"data/newsp/{n}"; os.makedirs(o, exist_ok=True); d.rename(columns={"max_ipSAE": "max_ipSAE"})[["uniprot_ac_1", "uniprot_ac_2", "passes", "string", "pdb", "max_ipSAE", "ipTM"]].to_csv(f"{o}/tiers.tsv.gz", sep="\t", index=False, compression="gzip")
    open(f"{o}/cand_proteins.fasta", "w").write("".join(f">{a}\n{seqs[a]}\n" for a in sorted(accs)))
    meta = dict(species=n, candidates=len(d), candidates_without_string_score=n_nostring, candidates_without_sequence=n_noseq, sequences_fetched_from_afdb_api=got, proteins=len(accs), s700=int((d.string >= 700).sum()), low_string_lt400_nopdb=int(((d.string < 400) & ~d.pdb).sum()), pdb_pairs=int(d.pdb.sum()), afm_confident=int(d.passes.sum()))
    json.dump(meta, open(f"{o}/tiers_meta.json", "w")); log(str(meta))
log("DONE")
