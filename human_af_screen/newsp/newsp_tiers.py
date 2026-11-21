"""Stage 2 inputs for the multi-species degree-matched cells (same construction as bern_univ/build_cand_negs2.py): per species, AFDB heterodimer candidates (STRING physical links, no threshold) with their
STRING v12 combined score (STRING ids mapped to UniProt accessions through the aliases file), PDB co-membership (SIFTS), AFM-confident flag, ipSAE, ipTM; sequences for every candidate protein (UniProt proteome FASTA, UniProt REST extras,
AlphaFold DB API for the rest). Candidates whose proteins have no sequence are dropped and counted. Writes data/<sp>/{tiers.tsv.gz, cand_proteins.fasta, tiers_meta.json}."""
import gzip, json, os, time, urllib.request, concurrent.futures as cf, numpy as np, pandas as pd
SP = {"soybean": 3847, "arabidopsis": 3702, "rat": 10116, "rice": 39947, "zebrafish": 7955}
T = pd.read_csv("../afdb/hetero_slim.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "tax_id_1", "tax_id_2", "ipTM", "max_ipSAE", "passes"])
def read(path):
    d, k = {}, None
    if not os.path.exists(path): return d
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
def string_file(tx, kind):
    p = f"string/{tx}.protein.{kind}.v12.0.txt.gz"
    if not os.path.exists(p): p = f"../string/{tx}.protein.{kind}.v12.0.txt.gz"
    return p
# SIFTS entry -> accessions (only accessions we need are filtered later)
entry_acc = {}
with gzip.open("../pdb/pdb_chain_uniprot.tsv.gz", "rt") as f:
    next(f); next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) > 2: entry_acc.setdefault(p[0], set()).add(p[2])
for n, tx in SP.items():
    t0 = time.time(); d = T[(T.tax_id_1 == tx) & (T.tax_id_2 == tx)].copy(); d = d[d.uniprot_ac_1 != d.uniprot_ac_2].drop_duplicates(["uniprot_ac_1", "uniprot_ac_2"])
    al = {}
    for l in gzip.open(string_file(tx, "aliases"), "rt"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2] == "UniProt_AC": al.setdefault(p[0], set()).add(p[1])
    sc = {}
    for l in gzip.open(string_file(tx, "physical.links.detailed"), "rt"):
        p = l.split()
        if p[0] == "protein1": continue
        for a in al.get(p[0], ()):
            for b in al.get(p[1], ()):
                if a != b: k = (a, b) if a < b else (b, a); s = int(p[-1]); sc[k] = max(sc.get(k, 0), s)
    key = [(a, b) if a < b else (b, a) for a, b in zip(d.uniprot_ac_1, d.uniprot_ac_2)]; d["string"] = [sc.get(k, np.nan) for k in key]; n_nostring = int(d.string.isna().sum()); d = d[d.string.notna()].copy()
    accs = set(d.uniprot_ac_1) | set(d.uniprot_ac_2); pdbpairs = set()
    for e, s in entry_acc.items():
        s2 = sorted(s & accs)
        if len(s2) > 1:
            for i in range(len(s2)):
                for j in range(i + 1, len(s2)): pdbpairs.add((s2[i], s2[j]))
    d["pdb"] = [k in pdbpairs for k in [(a, b) if a < b else (b, a) for a, b in zip(d.uniprot_ac_1, d.uniprot_ac_2)]]
    seqs = read(f"proteomes/{n}.fasta"); seqs.update(read(f"proteomes/{n}_extra.fasta")); miss = sorted(accs - set(seqs)); got = 0
    if miss:
        with cf.ThreadPoolExecutor(8) as ex, open(f"proteomes/{n}_extra.fasta", "a") as f:
            for a, s in ex.map(get, miss):
                if s: seqs[a] = s; f.write(f">{a}\n{s}\n"); got += 1
    ok = d.uniprot_ac_1.isin(seqs) & d.uniprot_ac_2.isin(seqs); n_noseq = int((~ok).sum()); d = d[ok]; accs = set(d.uniprot_ac_1) | set(d.uniprot_ac_2)
    o = f"data/{n}"; os.makedirs(o, exist_ok=True); d.rename(columns={"max_ipSAE": "max_ipSAE"})[["uniprot_ac_1", "uniprot_ac_2", "passes", "string", "pdb", "max_ipSAE", "ipTM"]].to_csv(f"{o}/tiers.tsv.gz", sep="\t", index=False, compression="gzip")
    open(f"{o}/cand_proteins.fasta", "w").write("".join(f">{a}\n{seqs[a]}\n" for a in sorted(accs)))
    meta = dict(species=n, candidates=len(d), candidates_without_string_score=n_nostring, candidates_without_sequence=n_noseq, sequences_fetched_from_afdb_api=got, proteins=len(accs), s700=int((d.string >= 700).sum()), low_string_lt400_nopdb=int(((d.string < 400) & ~d.pdb).sum()), pdb_pairs=int(d.pdb.sum()), afm_confident=int(d.passes.sum()))
    json.dump(meta, open(f"{o}/tiers_meta.json", "w")); print(meta, f"[{time.time() - t0:.0f} s]", flush=True)
