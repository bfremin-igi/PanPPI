"""Sensitivity to trimming (Fig 2c cells): keep ALL STRING>=700 positives and reweight the negatives instead of trimming the positives. Negatives = STRING-linked AFDB candidates with STRING<400 and no PDB co-membership (the pool the trimmed sets draw from).
Degree d = log(1+k_A)+log(1+k_B), k = protein degree in the evaluated (positive+negative) set; negatives in bin b of d (20 bins from the quantiles of the positives) get w_b = pi+_b/pi-_b (as for Fig 2d). Weighted AUPR at the resulting 1:1 balance
for PanPPI (3-seed mean blend), LoGoPPI, AFM ipSAE and ipTM, all proteins (L0). Compare with the trimmed cells (bern_univ/cand_results.csv, variant A). Output no_trim_reweight.csv."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AU
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); rows = []
def table(sp):
    if sp == "human":
        T = pd.read_csv(H + "bern_univ/candidates_afm.tsv.gz", sep="\t"); ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; ix = {a: k for k, a in enumerate(ids)}
        T["i"] = T.uniprot_ac_1.map(ix); T["j"] = T.uniprot_ac_2.map(ix); T = T.dropna(subset=["i", "j"]); T[["i", "j"]] = T[["i", "j"]].astype(int); a_, b_ = np.minimum(T.i, T.j), np.maximum(T.i, T.j); T["i"], T["j"] = a_, b_; T = T.drop_duplicates(["i", "j"])
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        Z = pd.DataFrame({"tri": M.tri.values, "pan": np.mean([z[f"pan_{s}_blend"] for s in S], axis=0), "lg": z["logoppi"] if "logoppi" in z.files else -z["logoppi_rank"]}); M = M.merge(Z, on="tri"); M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j)
        T = T.rename(columns={"i": "i2", "j": "j2"}).merge(M[["i2", "j2", "pan", "lg"]], on=["i2", "j2"], how="inner"); T["i"], T["j"] = T.i2, T.j2; return T
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz"); M["pan"] = np.mean([z[f"pan_{s}_blend"] for s in S], axis=0); M["lg"] = z["logoppi"] if "logoppi" in z.files else -z["logoppi_rank"]
    M["max_ipSAE"] = M["max_ipSAE"]; return M
for sp in ("human", "yeast", "mouse", "fly", "worm"):
    M = table(sp); sc = M["string"] if "string" in M else None; pdbc = M["pdb"].astype(bool) if "pdb" in M else False
    pos = M[(M["string"] >= 700)]; neg = M[(M["string"] < 400) & ~M["pdb"].astype(bool)]; E = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    k = pd.concat([E.i, E.j]).value_counts(); E["d"] = np.log1p(E.i.map(k).to_numpy()) + np.log1p(E.j.map(k).to_numpy())
    qs = np.quantile(E.d[E.y == 1], np.linspace(0, 1, 21)); qs[0], qs[-1] = -np.inf, np.inf; b = np.clip(np.searchsorted(qs, E.d, side="right") - 1, 0, 19); E["b"] = b
    pp = np.bincount(b[E.y == 1], minlength=20) / (E.y == 1).sum(); pn = np.bincount(b[E.y == 0], minlength=20) / max((E.y == 0).sum(), 1); w = np.where(E.y == 1, 1.0, np.where(pn[b] > 0, pp[b] / np.maximum(pn[b], 1e-12), 0.0)); w[E.y.to_numpy() == 0] *= (E.y == 1).sum() / w[E.y.to_numpy() == 0].sum()
    deg_auc = AU(E.y, E.d, sample_weight=w)
    for nm, col in (("PanPPI", "pan"), ("LoGoPPI", "lg"), ("AFM ipSAE", "max_ipSAE"), ("AFM ipTM", "ipTM")):
        if col not in E: continue
        rows.append(dict(species=sp, n_pos=int((E.y == 1).sum()), n_neg=int((E.y == 0).sum()), tool=nm, aupr_reweighted=AP(E.y, E[col], sample_weight=w), aupr_unweighted_raw=AP(E.y, E[col]), degree_only_auroc_after_weighting=deg_auc))
R = pd.DataFrame(rows); R.to_csv(H + "fig2/no_trim_reweight.csv", index=False); P = R.pivot(index="species", columns="tool", values="aupr_reweighted").round(3); print(P.to_string())
T = pd.read_csv(H + "bern_univ/cand_results.csv"); T = T[(T.tier == "s700") & (T.variant == "A") & (T.level == "L0") & (T.kept == True)].set_index("species")
print("\ntrimmed cells (L0, variant A) for comparison:"); print(T[["PanPPI blend aupr", "LoGoPPI aupr", "AFM ipSAE aupr", "AFM ipTM aupr"]].round(3).to_string())
