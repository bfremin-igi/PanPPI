"""Fig 2b intervals: retention of AFM-confident pairs in the top x% of ALL pairs, PanPPI (mean of 3 seeds) vs LoGoPPI, with a paired protein-clustered bootstrap (pairs assigned to their higher-degree protein within the AFM-confident set;
proteins resampled with replacement, 1,000 draws, both tools on the identical resample). Same ranks as univ_recall_table.py (rank fraction of each AFM-confident pair among all pairs of the proteome).
Human = pairs new to the model (strata 3-5). Output fig2/retention_ci.csv (set, f, pan, pan_lo, pan_hi, lg, lg_lo, lg_hi, diff, diff_lo, diff_hi, n)."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); FR = (0.01, 0.05, 0.10); rng = np.random.default_rng(11)
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; idx = {a: k for k, a in enumerate(ids)}
def load(sp):
    if sp == "human":
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        zz = pd.DataFrame({"tri": M.tri.values, **{f"b_{s}": z[f"pan_{s}_blend_rank"] for s in S}, "lg": z["logoppi_rank"]}); M = M.merge(zz, on="tri")
        T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "passes", "st"]); T["i"] = T.uniprot_ac_1.map(idx); T["j"] = T.uniprot_ac_2.map(idx); T = T.dropna(subset=["i", "j"]); T[["i", "j"]] = T[["i", "j"]].astype(int)
        a = np.minimum(T.i, T.j); b = np.maximum(T.i, T.j); T["i"], T["j"] = a, b; T = T.drop_duplicates(["i", "j"])[["i", "j", "passes", "st"]]
        M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j); M = M.merge(T.rename(columns={"i": "i2", "j": "j2"}), on=["i2", "j2"], how="inner"); return M[(M.passes.astype(bool)) & (M.st >= 3)]
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz")
    for s in S: M[f"b_{s}"] = z[f"pan_{s}_blend_rank"]
    M["lg"] = z["logoppi_rank"]; return M[M.passes.astype(bool)]
rows = []
for sp, lab in (("human", "human (clean)"), ("yeast", "yeast"), ("mouse", "mouse"), ("fly", "fly"), ("worm", "worm"), ("ecoli", "ecoli")):
    x = load(sp).reset_index(drop=True); deg = pd.concat([x.i, x.j]).value_counts(); g = np.where(x.i.map(deg) >= x.j.map(deg), x.i, x.j); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
    B = np.stack([x[f"b_{s}"].to_numpy() for s in S]); L = x.lg.to_numpy()
    def stat(ix, f): return np.mean([(B[k][ix] <= f).mean() for k in range(3)]), (L[ix] <= f).mean()
    for f in FR:
        p0, l0 = stat(np.arange(len(x)), f); bs = []
        for _ in range(1000):
            ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); bs.append(stat(ix, f))
        bs = np.array(bs); d = bs[:, 0] - bs[:, 1]
        rows.append(dict(set=lab, f=f, n=len(x), pan=p0, pan_lo=np.percentile(bs[:, 0], 2.5), pan_hi=np.percentile(bs[:, 0], 97.5), lg=l0, lg_lo=np.percentile(bs[:, 1], 2.5), lg_hi=np.percentile(bs[:, 1], 97.5), diff=p0 - l0, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
R = pd.DataFrame(rows); R.round(4).to_csv(H + "fig2/retention_ci.csv", index=False); print(R.round(3).to_string(index=False))
