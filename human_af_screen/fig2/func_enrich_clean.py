"""Panel g companion (v4): the same keyword-category recall as func_enrich.py (human, STRING>=700, variant A, level L0; calls at the 95th percentile of each replicate's negatives; AFM = confident flag) restricted to pairs NEW to the model:
strata 3-5 of candidates_with_strata (pair not in, and not homologous to, the human training pairs). Output func_enrich_clean.csv (keyword, n_clean, afm_clean, pan_clean, lg_clean)."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141")
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.tier == "s700") & (B.level == "L0")].copy(); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0)
P = B[B.label == 1].drop_duplicates(["i", "j"]).reset_index(drop=True); reps = sorted(set(B[B.label == 0].rep))
for nm in ("pan", "lg"):
    th = [np.quantile(B[(B.label == 0) & (B.rep == r)][nm], .95) for r in reps]; P[nm + "_call"] = np.mean([(P[nm] > t) for t in th], axis=0)
P["afm"] = P.afm_pass.astype(float); P["a"] = P.i.map(lambda x: ids[x]); P["b"] = P.j.map(lambda x: ids[x])
T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "st"]); T2 = T.rename(columns={"uniprot_ac_1": "b", "uniprot_ac_2": "a"}); T = pd.concat([T.rename(columns={"uniprot_ac_1": "a", "uniprot_ac_2": "b"}), T2]).drop_duplicates(["a", "b"]); P = P.merge(T, on=["a", "b"], how="left")
K = pd.read_csv(H + "fig2/annot/uniprot_human_keywords.tsv", sep="\t"); kw = {r.Entry: set(str(r.Keywords).split(";")) for r in K.itertuples()}; clean = (P.st >= 3).to_numpy(); Q = P[clean]; ss = [kw.get(a, set()) | kw.get(b, set()) for a, b in zip(Q.a, Q.b)]
G0 = pd.read_csv(H + "fig2/func_enrich.csv"); rows = []
for k in G0.keyword:
    m = np.array([k in s for s in ss]); rows.append(dict(keyword=k, n_clean=int(m.sum()), afm_clean=Q.afm[m].mean() if m.sum() else np.nan, pan_clean=Q.pan_call[m].mean() if m.sum() else np.nan, lg_clean=Q.lg_call[m].mean() if m.sum() else np.nan))
R = pd.DataFrame(rows); R.round(4).to_csv(H + "fig2/func_enrich_clean.csv", index=False); print(len(Q), "clean pairs;", int((R.n_clean >= 150).sum()), "of", len(R), "keywords with >= 150 clean pairs"); print(R.head(16).round(3).to_string(index=False))
