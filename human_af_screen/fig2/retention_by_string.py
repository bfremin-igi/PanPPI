"""Does retention depend on STRING support? Retention of AFM-confident pairs split by their STRING physical score (<400, 400-699, >=700), PanPPI (3-seed mean rank) vs LoGoPPI, same definitions as retention_extended.py. Protein-clustered bootstrap (300 draws; each pair -> its higher-degree protein). Output retention_by_string.csv."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); FR = (1e-3, 1e-2, 1e-1); rows = []; rng = np.random.default_rng(11)
ids_h = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; idx = {a: k for k, a in enumerate(ids_h)}
def load(sp):
    if sp == "human":
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        zz = pd.DataFrame({"tri": M.tri.values, **{f"b_{s}": z[f"pan_{s}_blend_rank"] for s in S}, "lg": z["logoppi_rank"]}); M = M.merge(zz, on="tri")
        T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "passes", "string", "st"]); T["i"] = T.uniprot_ac_1.map(idx); T["j"] = T.uniprot_ac_2.map(idx); T = T.dropna(subset=["i", "j"]); T[["i", "j"]] = T[["i", "j"]].astype(int)
        a_, b_ = np.minimum(T.i, T.j), np.maximum(T.i, T.j); T["i"], T["j"] = a_, b_; T = T.drop_duplicates(["i", "j"])[["i", "j", "passes", "string", "st"]]
        M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j); M = M.merge(T.rename(columns={"i": "i2", "j": "j2"}), on=["i2", "j2"], how="inner"); return M[M.passes.astype(bool) & (M.st >= 3)].reset_index(drop=True)   # new pairs only, as Fig 2b
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz")
    for s in S: M[f"b_{s}"] = z[f"pan_{s}_blend_rank"]
    M["lg"] = z["logoppi_rank"]; return M[M.passes.astype(bool)].reset_index(drop=True)
for sp, lab in (("human", "Human (new pairs)"), ("yeast", "Yeast"), ("mouse", "Mouse"), ("fly", "Fly"), ("worm", "Worm"), ("ecoli", "E. coli")):
    M = load(sp); M["tier"] = pd.cut(M.string, [-1, 399, 699, 1001], labels=["<400", "400-699", ">=700"])
    for t in ("<400", "400-699", ">=700"):
        K = M[M.tier == t]
        if len(K) < 30: continue
        dk = pd.concat([K.i, K.j]).value_counts(); g = np.where(K.i.map(dk) >= K.j.map(dk), K.i, K.j); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
        P = np.stack([K[f"b_{s}"].to_numpy() for s in S]); L = K.lg.to_numpy()
        for f in FR:
            bs = []
            for _ in range(300):
                ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); bs.append((np.mean([(P[k][ix] <= f).mean() for k in range(3)]), (L[ix] <= f).mean()))
            bs = np.array(bs); rows.append(dict(set=lab, string_tier=t, n_afm_confident=len(K), fraction=f, PanPPI=np.mean([(P[k] <= f).mean() for k in range(3)]), PanPPI_lo=np.percentile(bs[:, 0], 2.5), PanPPI_hi=np.percentile(bs[:, 0], 97.5), LoGoPPI=(L <= f).mean(), LoGoPPI_lo=np.percentile(bs[:, 1], 2.5), LoGoPPI_hi=np.percentile(bs[:, 1], 97.5)))
R = pd.DataFrame(rows); R["diff"] = R.PanPPI - R.LoGoPPI; R.round(4).to_csv(H + "fig2/retention_by_string.csv", index=False); pd.set_option("display.width", 200)
print(R[R.fraction == 1e-2][["set", "string_tier", "n_afm_confident", "PanPPI", "PanPPI_lo", "PanPPI_hi", "LoGoPPI", "LoGoPPI_lo", "LoGoPPI_hi"]].round(2).to_string(index=False))
