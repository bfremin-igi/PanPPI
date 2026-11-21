#!/usr/bin/env python3
"""Retention for the multi-species screen, with the same definitions as Fig 2b and the hub control. Rank fraction of an AFM-confident pair = share of the ALL-PAIRS SAMPLE (every pair among the 6,000-protein random subset) scoring at least as high;
retention at top x = share of AFM-confident pairs with rank fraction <= x. PanPPI = mean of the seeds available (blend; printed at the start), LoGoPPI = released model. Hub control: drop AFM-confident pairs that involve a protein in the top q of the
STRING-linked candidate degree distribution (q = 0, 1, 5, 10%). Protein-clustered bootstrap (each pair -> its higher-degree protein), 300 draws, PanPPI and LoGoPPI on the same resample. Output analysis_out/newsp/newsp_retention.csv."""
import glob, json, os, numpy as np, pandas as pd
import os
SPS = ["soybean", "arabidopsis", "rat", "rice", "zebrafish"]; SEEDS = tuple(sd for sd in ("s42", "s7", "s141") if all(os.path.exists(f"analysis_out/newsp/{sp}/panppi_{sd}.npz") for sp in SPS)); print("PanPPI seeds used:", SEEDS, flush=True); assert SEEDS, "no complete PanPPI seed"; FR = (1e-4, 1e-3, 1e-2, 1e-1); QS = (0.0, 0.01, 0.05, 0.10); rng = np.random.default_rng(5); rows = []
for sp in SPS:
    D = f"data/newsp/{sp}"; O = f"analysis_out/newsp/{sp}"; meta = json.load(open(D + "/meta.json")); NT = meta["universe_pairs"]
    Z = [np.load(f"{O}/panppi_{s}.npz", allow_pickle=True) for s in SEEDS]; a, b = Z[0]["afm_a"], Z[0]["afm_b"]
    for z in Z: assert (z["afm_a"] == a).all() and (z["afm_b"] == b).all()
    L = pd.read_csv(f"{O}/logoppi_afm.csv", dtype={"query": str, "text": str}); assert (L["query"].to_numpy() == a).all() and (L["text"].to_numpy() == b).all(); lg_afm = L["score"].to_numpy()
    lg_sub = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{O}/logoppi_subset/bp_*.npy"))]); assert len(lg_sub) == len(Z[0]["subset_blend"]), (len(lg_sub), len(Z[0]["subset_blend"]))
    sub = [np.sort(z["subset_blend"]) for z in Z]; lsub = np.sort(lg_sub); n = len(lsub)
    rank = lambda sorted_scores, x: 1 - np.searchsorted(sorted_scores, x, side="left") / n          # share of sample pairs scoring >= x
    pan_rank = np.stack([rank(sub[k], Z[k]["afm_blend"]) for k in range(len(SEEDS))]); lg_rank = rank(lsub, lg_afm)
    deg = pd.read_csv(D + "/cand_degree.tsv", sep="\t").set_index("protein").degree
    for q in QS:
        thr = np.quantile(deg.to_numpy(), 1 - q) if q > 0 else np.inf; hubs = set(deg.index[deg >= thr]) if q > 0 else set(); keep = np.array([(x not in hubs) and (y not in hubs) for x, y in zip(a, b)])
        da, db = deg.reindex(a[keep]).to_numpy(), deg.reindex(b[keep]).to_numpy(); g = np.where(da >= db, a[keep], b[keep]); u, inv = np.unique(g, return_inverse=True); groups = [np.where(inv == c)[0] for c in range(len(u))]
        P, Lr = pan_rank[:, keep], lg_rank[keep]
        for f in FR:
            rp = np.mean([(P[k] <= f).mean() for k in range(len(SEEDS))]); rl = (Lr <= f).mean(); bs = []
            for _ in range(300):
                ix = np.concatenate([groups[c] for c in rng.integers(0, len(u), len(u))]); bs.append((np.mean([(P[k][ix] <= f).mean() for k in range(len(SEEDS))]), (Lr[ix] <= f).mean()))
            bs = np.array(bs); d = bs[:, 0] - bs[:, 1]; nl = int(keep.sum())
            rows.append(dict(species=sp, hub_fraction_removed=q, n_afm_confident=nl, universe_proteins=meta["universe_proteins"], universe_pairs=NT, fraction=f, pairs_sent=f * NT, seeds=len(SEEDS), PanPPI=rp, PanPPI_lo=np.percentile(bs[:, 0], 2.5), PanPPI_hi=np.percentile(bs[:, 0], 97.5),
                             LoGoPPI=rl, LoGoPPI_lo=np.percentile(bs[:, 1], 2.5), LoGoPPI_hi=np.percentile(bs[:, 1], 97.5), diff=rp - rl, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                             pairs_per_hit_PanPPI=f * NT / max(rp * nl, 1e-9), pairs_per_hit_random=NT / nl))
R = pd.DataFrame(rows); R.to_csv("analysis_out/newsp/newsp_retention.csv", index=False); pd.set_option("display.width", 230)
print(R[(R.hub_fraction_removed == 0) & R.fraction.isin([1e-3, 1e-2, 1e-1])][["species", "n_afm_confident", "fraction", "PanPPI", "LoGoPPI", "diff", "diff_lo", "diff_hi", "pairs_per_hit_PanPPI", "pairs_per_hit_random"]].round(3).to_string(index=False))
print(R[(R.fraction == 1e-2)][["species", "hub_fraction_removed", "n_afm_confident", "PanPPI", "LoGoPPI", "diff", "diff_lo", "diff_hi"]].round(3).to_string(index=False))
