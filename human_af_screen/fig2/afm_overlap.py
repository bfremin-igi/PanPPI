"""Was AFM likely to have SEEN the direct-contact pairs? (user: release date != novelty). For each human direct-contact pair (pdb_contact_full.tsv, >=3 residues each):
 A = some PDB entry released <= CUT contains BOTH proteins (exact pair in the pre-cutoff PDB; SIFTS UniProt-chain map, any entry size)
 B = not A, but each protein has a pre-cutoff structure in some entry
 C = at least one protein has NO pre-cutoff structure.  Homologs are NOT tested here (needs sequence search against pre-cutoff PDB).
CUT = 2021-09-30 (my recollection of the AF-Multimer v2.x/v3 training cutoff; not verified). Entry release dates: RCSB GraphQL (public)."""
import gzip, json, collections, urllib.request, numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); CUT = "2021-09-30"
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
C = pd.read_csv(H + "fig2/pdb_contact_full.tsv", sep="\t"); C = C[(C.na >= 3) & (C.nb >= 3)]; D = C[["i", "j"]].drop_duplicates().reset_index(drop=True)
D["a"] = D.i.map(lambda x: ids[x]); D["b"] = D.j.map(lambda x: ids[x]); need = set(D.a) | set(D.b); ent = collections.defaultdict(set)
with gzip.open(H + "pdb/pdb_chain_uniprot.tsv.gz", "rt") as f:
    next(f); next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) > 2 and p[2] in need: ent[p[2]].add(p[0].upper())
alle = sorted(set().union(*ent.values())); print("proteins", len(need), "entries to date", len(alle), flush=True); date = {}
for k in range(0, len(alle), 400):
    q = '{entries(entry_ids:%s){rcsb_id rcsb_accession_info{initial_release_date}}}' % json.dumps(alle[k:k + 400])
    r = urllib.request.Request("https://data.rcsb.org/graphql", data=json.dumps({"query": q}).encode(), headers={"Content-Type": "application/json"})
    for e in json.load(urllib.request.urlopen(r, timeout=120))["data"]["entries"]:
        if e: date[e["rcsb_id"]] = e["rcsb_accession_info"]["initial_release_date"][:10]
print("dated", len(date), flush=True)
pre = lambda p: {e for e in ent[p] if date.get(e, "9999") <= CUT}
cat = []
for a, b in zip(D.a, D.b):
    both = pre(a) & pre(b)
    cat.append("A exact pair pre-cutoff" if both else ("B both proteins seen separately" if pre(a) and pre(b) else "C a protein has no pre-cutoff structure"))
D["cat"] = cat
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.tier == "pdb") & (B.level == "L0")].copy(); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0)
P = B[B.label == 1].drop_duplicates(["i", "j"]).merge(D[["i", "j", "cat"]], on=["i", "j"]); N = B[B.label == 0].drop_duplicates(["i", "j"])
th = {k: np.quantile(N[c], .95) for k, c in (("PanPPI", "pan"), ("LoGoPPI", "lg"), ("AFM ipSAE", "ipsae"), ("AFM ipTM", "iptm"))}
rows = []
for c_, g in P.groupby("cat"): rows.append(dict(category=c_, n=len(g), share=round(len(g) / len(P), 3), **{k: round(float((g[col] > th[k]).mean()), 3) for k, col in (("PanPPI", "pan"), ("LoGoPPI", "lg"), ("AFM ipSAE", "ipsae"), ("AFM ipTM", "iptm"))}))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(H + "fig2/afm_overlap.csv", index=False)
