"""Candidate highlight pairs for Fig 2 that are FAIR to AlphaFold-Multimer and that only PanPPI calls.
Human PDB-tier positives (variant A, L0 of bern_univ/out3/human) with a contact table row (pdb_contact_full.tsv). Criteria, all stated before looking at the list:
  structure: entry has <= 3 chains; >= 12 residues of EACH protein within 4.5 A of the partner (a real interface, not a peptide/edge contact);
             the chains cover >= 50% of the FULL-LENGTH UniProt sequence of BOTH proteins (so AFM's full-length input matches the structure);
  evidence : STRING physical >= 900;
  tools    : PanPPI blend above the 95th percentile of the negatives in ALL 3 replicates; LoGoPPI and ProteomeLM below their 95th percentile in all 3 replicates;
             AFM flag 0, ipTM < 0.5, ipSAE < 0.3.
Output: example_candidates.csv sorted by PanPPI percentile among the negatives."""
import gzip, collections, numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; SEEDS = ("s42", "s7", "s141")
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.level == "L0") & (B.tier == "pdb")].copy()
B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in SEEDS], axis=0)
PM = pd.read_csv(H + "bern_univ/out3/human/pm_scores.tsv.gz", sep="\t"); B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left")
P = B[B.label == 1].drop_duplicates(["i", "j"]).reset_index(drop=True); reps = sorted(set(B[B.label == 0].rep))
for nm in ("pan", "lg", "pm"):
    th = [np.quantile(B[(B.label == 0) & (B.rep == r)][nm].dropna(), .95) for r in reps]; P[nm + "_calls"] = sum((P[nm] > t).astype(int) for t in th)
    negs = np.sort(B[(B.label == 0)][nm].dropna().to_numpy()); P[nm + "_pct"] = np.searchsorted(negs, P[nm].to_numpy()) / len(negs) * 100
C = pd.read_csv(H + "fig2/pdb_contact_full.tsv", sep="\t"); C = C[C.na >= 0]; C["minc"] = np.minimum(C.na, C.nb)
C = C.sort_values("minc", ascending=False).drop_duplicates(["i", "j"])   # best entry per pair
P = P.merge(C, on=["i", "j"], how="inner")
seq = {}; name = None
for l in open(H + "seqs/screen_proteins.fasta"):
    if l.startswith(">"): name = l[1:].split()[0].split("|")[1] if "|" in l.split()[0] else l[1:].split()[0]; seq[name] = 0
    else: seq[name] += len(l.strip())
cov = collections.defaultdict(lambda: collections.defaultdict(list))
with gzip.open(H + "pdb/pdb_chain_uniprot.tsv.gz", "rt") as f:
    next(f); next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) > 8 and p[0] in set(P.entry): cov[(p[0], p[1], p[2])]["r"].append((int(p[7]), int(p[8])))
def covered(e, ch, acc):
    r = cov.get((e, ch, acc), {}).get("r", []); return sum(b - a + 1 for a, b in r) / seq[acc] if r and acc in seq else np.nan
P["a"] = P.i.map(lambda x: ids[x]); P["b"] = P.j.map(lambda x: ids[x])
P["cov_a"] = [covered(e, c, a) for e, c, a in zip(P.entry, P.chainA, P.a)]; P["cov_b"] = [covered(e, c, b) for e, c, b in zip(P.entry, P.chainB, P.b)]
print("pairs with a contact row:", len(P), "| direct (>=3 each):", int((P.minc >= 3).sum()))
S = P[(P.nch <= 3) & (P.minc >= 12) & (P.cov_a >= .5) & (P.cov_b >= .5) & (P.string >= 900) & (P.pan_calls == 3) & (P.lg_calls == 0) & (P.pm_calls == 0) & (P.afm_pass == 0) & (P.iptm < .5) & (P.ipsae < .3)]
print("pass all criteria:", len(S))
K = pd.read_csv(H + "fig2/annot/uniprot_human_keywords.tsv", sep="\t").set_index("Entry")
S = S.assign(genes=[f"{K['Gene Names (primary)'].get(a, a)}-{K['Gene Names (primary)'].get(b, b)}" for a, b in zip(S.a, S.b)], kw=[";".join(sorted((set(str(K.Keywords.get(a, "")).split(";")) | set(str(K.Keywords.get(b, "")).split(";"))) & {"Kinase", "Apoptosis", "Cell cycle", "DNA damage", "DNA repair", "Ubl conjugation pathway", "Immunity", "Receptor", "Transcription regulation", "Host-virus interaction", "Hydrolase", "Proto-oncogene", "Tumor suppressor"})) for a, b in zip(S.a, S.b)])
cols = ["genes", "a", "b", "entry", "nch", "na", "nb", "cov_a", "cov_b", "string", "pan_pct", "lg_pct", "pm_pct", "iptm", "ipsae", "kw"]
S = S.sort_values("pan_pct", ascending=False); S[cols].round(3).to_csv(H + "fig2/example_candidates.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(S[cols].round(2).head(40).to_string(index=False))
