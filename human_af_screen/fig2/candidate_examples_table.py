"""All pairs that satisfy the stated example-selection rule (Methods, 'Selection of examples'): human PDB-tier direct contacts with an interface of >= 10 residues per protein in an entry of <= 4 chains,
STRING physical >= 700, no AFM high-confidence call and ipTM < 0.5, PanPPI (3-seed mean of global+MaxSim blend) above the 95th percentile of the negatives in >= 2 of 3 replicates and LoGoPPI in <= 1.
Also reports ProteomeLM-PPI, ipSAE and the training-overlap stratum (1 exact pair in training, 2 homologous training positive pair, 3-5 new). Output candidate_examples_table.csv."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141")
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.level == "L0") & (B.tier == "pdb")].copy()
B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0)
PM = pd.read_csv(H + "bern_univ/out3/human/pm_scores.tsv.gz", sep="\t"); B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left")
P = B[B.label == 1].drop_duplicates(["i", "j"]).reset_index(drop=True); reps = sorted(set(B[B.label == 0].rep))
for nm in ("pan", "lg", "pm"):
    th = [np.quantile(B[(B.label == 0) & (B.rep == r)][nm].dropna(), .95) for r in reps]; P[nm + "_calls"] = sum((P[nm] > t).astype(int) for t in th)
    negs = np.sort(B[B.label == 0][nm].dropna().to_numpy()); P[nm + "_pct"] = np.searchsorted(negs, P[nm].to_numpy()) / len(negs) * 100
C = pd.read_csv(H + "fig2/pdb_contact_full.tsv", sep="\t"); C = C[C.na >= 0]; C = C[(C.nch <= 4) & (C.na >= 10) & (C.nb >= 10)].copy(); C["minc"] = np.minimum(C.na, C.nb)
nent = C.groupby(["i", "j"]).entry.nunique().rename("n_qualifying_entries").reset_index(); C = C.sort_values("minc", ascending=False).drop_duplicates(["i", "j"]).merge(nent, on=["i", "j"])
P = P.merge(C, on=["i", "j"], how="inner"); P["a"] = P.i.map(lambda x: ids[x]); P["b"] = P.j.map(lambda x: ids[x])
T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "st"]); T = pd.concat([T, T.rename(columns={"uniprot_ac_1": "uniprot_ac_2", "uniprot_ac_2": "uniprot_ac_1"})]).drop_duplicates(["uniprot_ac_1", "uniprot_ac_2"])
P = P.merge(T.rename(columns={"uniprot_ac_1": "a", "uniprot_ac_2": "b", "st": "overlap_stratum"}), on=["a", "b"], how="left")
n0 = len(P); R = P[(P.string >= 700) & (P.afm_pass == 0) & (P.iptm < .5) & (P.pan_calls >= 2) & (P.lg_calls <= 1)].copy()
K = pd.read_csv(H + "fig2/annot/uniprot_human_keywords.tsv", sep="\t").set_index("Entry")
R["gene_a"] = R.a.map(lambda x: K["Gene Names (primary)"].get(x, x)); R["gene_b"] = R.b.map(lambda x: K["Gene Names (primary)"].get(x, x))
R["new_to_panppi"] = R.overlap_stratum >= 3
out = R.rename(columns={"entry": "pdb_entry", "nch": "chains_in_entry", "na": "interface_residues_a", "nb": "interface_residues_b", "string": "string_physical", "iptm": "afm_ipTM", "ipsae": "afm_ipSAE", "pan_pct": "panppi_percentile_among_negatives", "lg_pct": "logoppi_percentile", "pm_pct": "proteomelm_percentile", "pan_calls": "panppi_calls_of_3", "lg_calls": "logoppi_calls_of_3"})
cols = ["gene_a", "gene_b", "a", "b", "pdb_entry", "chains_in_entry", "interface_residues_a", "interface_residues_b", "n_qualifying_entries", "string_physical", "afm_ipTM", "afm_ipSAE", "panppi_percentile_among_negatives", "panppi_calls_of_3", "logoppi_percentile", "logoppi_calls_of_3", "proteomelm_percentile", "overlap_stratum", "new_to_panppi"]
out = out[cols].sort_values(["new_to_panppi", "panppi_percentile_among_negatives"], ascending=[False, False]).round(3); out.to_csv(H + "fig2/candidate_examples_table.csv", index=False)
print(f"human PDB-tier direct-contact pairs with qualifying interface/entry: {n0}; pass the full rule: {len(out)}; of which new to PanPPI (stratum 3-5): {int(out.new_to_panppi.sum())}")
for k in ("2kbw", "2le8"): print(k, "in table:", bool((out.pdb_entry == k).any()), "| best-entry rows:", out[out.pdb_entry == k][["gene_a", "gene_b", "afm_ipTM", "panppi_calls_of_3", "logoppi_calls_of_3", "overlap_stratum"]].to_dict("records"))
print(out.head(12).to_string(index=False))
