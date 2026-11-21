"""Data for the supplementary figure on Figure 2 (retention of AFM-confident pairs): (a) other AFM-confidence thresholds, (d) the three PanPPI seeds separately, (e) pairs the model has not seen in training.
Same definitions as fig2/retention_extended.py: rank among ALL pairs of the proteome (blend rank, mean of seeds), retention at top x = share of the evaluated pairs with rank <= x; human = pairs new to the model (st >= 3) except in panel (e);
protein-clustered bootstrap (each pair -> its higher-degree protein within the evaluated set), 200 draws, PanPPI and LoGoPPI on the same resample. 'Homolog' = a human training protein at >= 40% identity and >= 50% query coverage (maxid_cov50 >= 0.40).
Outputs supp2_threshold.csv, supp2_seeds.csv, supp2_novelty.csv in this folder."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141"); FR = (1e-2, 1e-1); rng = np.random.default_rng(31); NB = 200
mx_h = pd.read_csv(H + "homology/maxid.tsv", sep="\t").set_index("query").maxid_cov50; mx_u = pd.read_csv(H + "fig2/supp2/maxid_univ.tsv", sep="\t").set_index("query").maxid_cov50
ids_h = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; idx = {a: k for k, a in enumerate(ids_h)}
def load(sp):
    if sp == "human":
        M = pd.read_csv(H + "univ_extract/meta.tsv.gz", sep="\t", usecols=["tri", "i", "j"]); z = np.load(H + "univ_extract/scores.npz")
        zz = pd.DataFrame({"tri": M.tri.values, **{f"b_{s}": z[f"pan_{s}_blend_rank"] for s in S}, "lg": z["logoppi_rank"]}); M = M.merge(zz, on="tri")
        T = pd.read_csv(H + "bern_univ/candidates_afm.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "passes", "max_ipSAE", "ipTM", "st"]); T["i"] = T.uniprot_ac_1.map(idx); T["j"] = T.uniprot_ac_2.map(idx); T = T.dropna(subset=["i", "j"])
        T["i"], T["j"] = np.minimum(T.i, T.j).astype(int), np.maximum(T.i, T.j).astype(int); T = T.drop_duplicates(["i", "j"])[["i", "j", "passes", "max_ipSAE", "ipTM", "st"]]
        M["i2"], M["j2"] = np.minimum(M.i, M.j), np.maximum(M.i, M.j); M = M.merge(T.rename(columns={"i": "i2", "j": "j2"}), on=["i2", "j2"], how="inner").reset_index(drop=True)
        pid = np.array(ids_h); M["pa"], M["pb"] = pid[M.i2.values], pid[M.j2.values]; M["mxa"], M["mxb"] = M.pa.map(mx_h).values, M.pb.map(mx_h).values
        return M
    M = pd.read_csv(H + f"univ_nonhuman/{sp}/truth_rows_meta.tsv.gz", sep="\t"); z = np.load(H + f"univ_nonhuman/{sp}/truth_rows_scores.npz")
    for s in S: M[f"b_{s}"] = z[f"pan_{s}_blend_rank"]
    M["lg"] = z["logoppi_rank"]; ids = np.array([l.strip() for l in open(H + f"univ/{sp}/ids.txt") if l.strip()]); M["pa"], M["pb"] = ids[M.i.values], ids[M.j.values]; M["mxa"], M["mxb"] = M.pa.map(mx_u).values, M.pb.map(mx_u).values
    M["st"] = np.nan; return M
def retention(K, f):
    P = np.stack([K[f"b_{s}"].to_numpy() for s in S]); return [(p <= f).mean() for p in P], (K.lg.to_numpy() <= f).mean()
def boot(K, f):
    dk = pd.concat([K.pa, K.pb]).value_counts(); g = np.where(K.pa.map(dk).to_numpy() >= K.pb.map(dk).to_numpy(), K.pa.to_numpy(), K.pb.to_numpy()); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
    P = np.stack([K[f"b_{s}"].to_numpy() for s in S]); L = K.lg.to_numpy(); out = []
    for _ in range(NB):
        ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); out.append((np.mean([(p[ix] <= f).mean() for p in P]), (L[ix] <= f).mean()))
    o = np.array(out); return np.percentile(o[:, 0], [2.5, 97.5]), np.percentile(o[:, 1], [2.5, 97.5]), np.percentile(o[:, 0] - o[:, 1], [2.5, 97.5])
SP = [("human", "Human (new pairs)"), ("yeast", "Yeast"), ("mouse", "Mouse"), ("fly", "Fly"), ("worm", "Worm"), ("ecoli", "E. coli")]
thr, seeds_rows, nov = [], [], []
for sp, lab in SP:
    M = load(sp); base = M[M.st >= 3] if sp == "human" else M
    sets = {"paper label (ipSAE >= 0.6 and pDockQ2 >= 0.23)": base[base.passes.astype(bool)]}
    for t in (0.5, 0.7, 0.8): sets[f"ipSAE >= {t}"] = base[base.max_ipSAE >= t]
    sets["ipTM >= 0.7"] = base[base.ipTM >= 0.7]
    for name, K in sets.items():
        if len(K) < 30: continue
        for f in FR:
            rp, rl = retention(K, f); (plo, phi), (llo, lhi), (dlo, dhi) = boot(K, f) if f == 1e-2 else ((np.nan,) * 2, (np.nan,) * 2, (np.nan,) * 2)
            thr.append(dict(species=lab, definition=name, n=len(K), fraction=f, PanPPI=np.mean(rp), PanPPI_lo=plo, PanPPI_hi=phi, LoGoPPI=rl, LoGoPPI_lo=llo, LoGoPPI_hi=lhi, diff_lo=dlo, diff_hi=dhi))
    K = sets["paper label (ipSAE >= 0.6 and pDockQ2 >= 0.23)"]
    for f in FR:
        rp, rl = retention(K, f); seeds_rows.append(dict(species=lab, fraction=f, n=len(K), **{f"PanPPI_{s}": r for s, r in zip(S, rp)}, LoGoPPI=rl))
    # novelty: AFM-confident (flag) pairs; human all strata
    A = M[M.passes.astype(bool)].copy(); A["hom_a"] = A.mxa.fillna(0) >= 0.4; A["hom_b"] = A.mxb.fillna(0) >= 0.4; A["nhom"] = A.hom_a.astype(int) + A.hom_b.astype(int)
    if sp == "human": groups = {"pair or homologous pair in training": A[A.st <= 2], "new pair, both proteins have a training homolog": A[(A.st >= 3) & (A.nhom == 2)], "new pair, one protein without a homolog": A[(A.st >= 3) & (A.nhom == 1)], "new pair, neither protein has a homolog": A[(A.st >= 3) & (A.nhom == 0)]}
    else: groups = {"at least one protein has a training homolog": A[A.nhom >= 1], "neither protein has a training homolog": A[A.nhom == 0]}
    for name, K in groups.items():
        for f in FR:
            if len(K) < 30: nov.append(dict(species=lab, group=name, n=len(K), fraction=f)); continue
            rp, rl = retention(K, f); (plo, phi), (llo, lhi), (dlo, dhi) = boot(K, f)
            nov.append(dict(species=lab, group=name, n=len(K), fraction=f, PanPPI=np.mean(rp), PanPPI_lo=plo, PanPPI_hi=phi, LoGoPPI=rl, LoGoPPI_lo=llo, LoGoPPI_hi=lhi, diff_lo=dlo, diff_hi=dhi))
    print(lab, "done", flush=True)
pd.DataFrame(thr).round(4).to_csv("supp2_threshold.csv", index=False); pd.DataFrame(seeds_rows).round(4).to_csv("supp2_seeds.csv", index=False); pd.DataFrame(nov).round(4).to_csv("supp2_novelty.csv", index=False)
