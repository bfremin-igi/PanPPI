"""Hub control for retention (Fig 2b): drop AFM-confident pairs that involve a hub, then recompute retention of the remainder, PanPPI (3-seed mean) vs LoGoPPI.
Degree = number of STRING-linked AFDB candidate partners of a protein (the candidate graph = every row of the truth file, confident or not). Hubs = proteins in the top HUB_Q fraction of that degree distribution
(default 5%, also 10% and 1%). Protein-clustered bootstrap as in retention_extended.py (each pair -> higher-degree protein), 300 draws. Output retention_hub_trim.csv."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); FR = (1e-3, 1e-2, 1e-1); QS = (0.0, 0.01, 0.05, 0.10); rng = np.random.default_rng(5)
ids_h = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; idx = {a: k for k, a in enumerate(ids_h)}
def load(sp):
    if sp == "human (clean)":
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        zz = pd.DataFrame({"tri": M.tri.values, **{f"b_{s}": z[f"pan_{s}_blend_rank"] for s in S}, "lg": z["logoppi_rank"]}); M = M.merge(zz, on="tri")
        T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "passes", "st"]); T["i"] = T.uniprot_ac_1.map(idx); T["j"] = T.uniprot_ac_2.map(idx); T = T.dropna(subset=["i", "j"]); T[["i", "j"]] = T[["i", "j"]].astype(int)
        a_, b_ = np.minimum(T.i, T.j), np.maximum(T.i, T.j); T["i"], T["j"] = a_, b_; T = T.drop_duplicates(["i", "j"])[["i", "j", "passes", "st"]]
        cand = T[["i", "j"]]                                              # the whole human candidate graph
        M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j); M = M.merge(T.rename(columns={"i": "i2", "j": "j2"}), on=["i2", "j2"], how="inner"); M = M[M.passes.astype(bool) & (M.st >= 3)]
        return M.reset_index(drop=True), cand
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz")
    for s in S: M[f"b_{s}"] = z[f"pan_{s}_blend_rank"]
    M["lg"] = z["logoppi_rank"]; cand = M[["i", "j"]]; return M[M.passes.astype(bool)].reset_index(drop=True), cand
rows = []
for sp, lab in (("human (clean)", "Human (new pairs)"), ("yeast", "Yeast"), ("mouse", "Mouse"), ("fly", "Fly"), ("worm", "Worm"), ("ecoli", "E. coli")):
    M, cand = load(sp); deg = pd.concat([cand.i, cand.j]).value_counts()
    for q in QS:
        thr = np.quantile(deg.to_numpy(), 1 - q) if q > 0 else np.inf; hubs = set(deg.index[deg >= thr]) if q > 0 else set()
        keep = ~(M.i.isin(hubs) | M.j.isin(hubs)).to_numpy(); K = M[keep].reset_index(drop=True); n = len(K)
        dk = pd.concat([K.i, K.j]).value_counts(); g = np.where(K.i.map(dk) >= K.j.map(dk), K.i, K.j); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
        pan = np.stack([K[f"b_{s}"].to_numpy() for s in S]); lg = K.lg.to_numpy()
        for f in FR:
            rp = np.mean([(pan[k] <= f).mean() for k in range(3)]); rl = (lg <= f).mean(); bs = []
            for _ in range(300):
                ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); bs.append((np.mean([(pan[k][ix] <= f).mean() for k in range(3)]), (lg[ix] <= f).mean()))
            bs = np.array(bs); d = bs[:, 0] - bs[:, 1]
            rows.append(dict(set=lab, hub_fraction_removed=q, hub_degree_cutoff=(float(thr) if q > 0 else np.nan), n_afm_confident_left=n, fraction=f, PanPPI=rp, LoGoPPI=rl, diff=rp - rl, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
R = pd.DataFrame(rows); R.to_csv(H + "fig2/retention_hub_trim.csv", index=False); pd.set_option("display.width", 220)
print(R[R.fraction == 0.01].round(3).to_string(index=False)); print(); print(R[R.fraction == 0.001][["set", "hub_fraction_removed", "n_afm_confident_left", "PanPPI", "LoGoPPI", "diff", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
