"""EXPLORATORY (not in the paper): retention of AFM-confident pairs at tighter shortlists (0.01%, 0.1%, 1%, 10% of ALL pairs of the proteome), with absolute pair counts and a compute-yield table, and simple baselines.
Baselines: random (retention = fraction), protein-length product (rank by product of the two sequence lengths; rank fractions estimated on 3M random pairs), PanPPI global branch alone, PanPPI MaxSim branch alone, LoGoPPI. PanPPI = mean over three seeds of the per-seed retention.
Universe size P = N(N-1)/2 with N = proteins of the species file. AFM-confident pairs are known only among STRING-linked candidates, so yields are lower bounds. Output fig2/retention_extended.csv."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); FR = (1e-4, 1e-3, 1e-2, 1e-1); rng = np.random.default_rng(21)
def fasta_len(path):
    L = {}; k = None
    for l in open(path):
        l = l.rstrip()
        if l.startswith(">"): k = l[1:].split()[0]; L[k] = 0
        elif k: L[k] += len(l)
    return L
ids_h = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; idx = {a: k for k, a in enumerate(ids_h)}
def load(sp):
    if sp in ("human", "human (clean)"):
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        zz = pd.DataFrame({"tri": M.tri.values, **{f"b_{s}": z[f"pan_{s}_blend_rank"] for s in S}, **{f"g_{s}": z[f"pan_{s}_global_rank"] for s in S}, **{f"m_{s}": z[f"pan_{s}_maxsim_rank"] for s in S}, "lg": z["logoppi_rank"]}); M = M.merge(zz, on="tri")
        T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "passes", "st"]); T["i"] = T.uniprot_ac_1.map(idx); T["j"] = T.uniprot_ac_2.map(idx); T = T.dropna(subset=["i", "j"]); T[["i", "j"]] = T[["i", "j"]].astype(int)
        a_, b_ = np.minimum(T.i, T.j), np.maximum(T.i, T.j); T["i"], T["j"] = a_, b_; T = T.drop_duplicates(["i", "j"])[["i", "j", "passes", "st"]]
        M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j); M = M.merge(T.rename(columns={"i": "i2", "j": "j2"}), on=["i2", "j2"], how="inner"); M = M[M.passes.astype(bool)]
        if sp == "human (clean)": M = M[M.st >= 3]
        L = fasta_len(H + "seqs/screen_proteins.fasta"); lens = np.array([L[a] for a in ids_h], float); return M.reset_index(drop=True), lens
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz")
    for s in S: M[f"b_{s}"] = z[f"pan_{s}_blend_rank"]; M[f"g_{s}"] = z[f"pan_{s}_global_rank"]; M[f"m_{s}"] = z[f"pan_{s}_maxsim_rank"]
    M["lg"] = z["logoppi_rank"]; M = M[M.passes.astype(bool)].reset_index(drop=True)
    ids = [l.strip() for l in open(H + f"univ/{sp}/ids.txt") if l.strip()]; L = fasta_len(H + f"univ/{sp}/proteins.fasta"); miss = [a for a in ids if a not in L]; assert not miss, (sp, miss[:3]); return M, np.array([L[a] for a in ids], float)
rows = []
for sp, lab, N in (("human (clean)", "Human (new pairs)", 19408), ("human", "Human (all)", 19408), ("yeast", "Yeast", 5960), ("mouse", "Mouse", 20612), ("fly", "Fly", 12786), ("worm", "Worm", 19734), ("ecoli", "E. coli", 4091)):
    M, lens = load(sp); n = len(M); P = N * (N - 1) // 2
    # length-product rank fraction among all pairs (estimate on 3M random pairs)
    a = rng.integers(0, N, 3_000_000); b = rng.integers(0, N, 3_000_000); keep = a != b; sc = np.sort((lens[a] * lens[b])[keep]); ps = lens[M.i.to_numpy()] * lens[M.j.to_numpy()]; lenrank = 1 - np.searchsorted(sc, ps, side="left") / len(sc)
    deg = pd.concat([M.i, M.j]).value_counts(); g = np.where(M.i.map(deg) >= M.j.map(deg), M.i, M.j); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
    arr = {"PanPPI blend": np.stack([M[f"b_{s}"].to_numpy() for s in S]), "PanPPI global only": np.stack([M[f"g_{s}"].to_numpy() for s in S]), "PanPPI MaxSim only": np.stack([M[f"m_{s}"].to_numpy() for s in S]), "LoGoPPI": M.lg.to_numpy()[None], "Length product": lenrank[None]}
    for f in FR:
        row = dict(set=lab, n_afm_confident=n, universe_pairs=P, fraction=f, pairs_sent=f * P)
        for nm, A in arr.items(): row[nm] = np.mean([(A[k] <= f).mean() for k in range(len(A))])
        bs = []
        for _ in range(500):
            ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); bs.append((np.mean([(arr["PanPPI blend"][k][ix] <= f).mean() for k in range(3)]), (arr["LoGoPPI"][0][ix] <= f).mean()))
        bs = np.array(bs); row["PanPPI blend lo"], row["PanPPI blend hi"] = np.percentile(bs[:, 0], [2.5, 97.5]); row["LoGoPPI lo"], row["LoGoPPI hi"] = np.percentile(bs[:, 1], [2.5, 97.5]); d = bs[:, 0] - bs[:, 1]; row["diff Pan-Lg"] = row["PanPPI blend"] - row["LoGoPPI"]; row["diff lo"], row["diff hi"] = np.percentile(d, [2.5, 97.5])
        row["afm_found_pan"] = row["PanPPI blend"] * n; row["pairs_per_hit_pan"] = row["pairs_sent"] / max(row["afm_found_pan"], 1e-9); row["pairs_per_hit_random"] = P / n; rows.append(row)
R = pd.DataFrame(rows); R.to_csv(H + "fig2/retention_extended.csv", index=False)
pd.set_option("display.width", 250); print(R[["set", "n_afm_confident", "fraction", "pairs_sent", "PanPPI blend", "PanPPI global only", "PanPPI MaxSim only", "LoGoPPI", "Length product", "diff Pan-Lg", "diff lo", "diff hi", "pairs_per_hit_pan", "pairs_per_hit_random"]].round(3).to_string(index=False))
