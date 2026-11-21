"""Panel f second view: force PanPPI and LoGoPPI to call exactly as many pairs as the AFM-confident flag (per cell, per replicate): precision and true pairs found, vs AFM itself. Same kept cells as eval_cand.py (variant A, not L1)."""
import json, numpy as np, pandas as pd
SEEDS = ("s42", "s7", "s141"); R = pd.read_csv("cand_results.csv"); keptc = {(r.species, r.tier, r.level) for r in R[(R.variant == "A") & R.kept & (R.level != "L1")].itertuples()}; rows = []
for sp in ["human", "ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"]:
    B = pd.read_csv(f"out3/{sp}/cand_scores.tsv.gz", sep="\t"); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in SEEDS], axis=0)
    for (tier, lvl), _ in B[B.variant == "A"].groupby(["tier", "level"]):
        if (sp, tier, lvl) not in keptc: continue
        G = B[(B.variant == "A") & (B.tier == tier) & (B.level == lvl)]; P = G[G.label == 1]; acc = {k: [] for k in ("pan", "lg", "afm")}
        for r in sorted(set(G[G.label == 0].rep)):
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); D = pd.concat([P.sample(k, random_state=r + 1) if len(P) > k else P, N.sample(k, random_state=r + 7) if len(N) > k else N]); y = D.label.to_numpy(); kk = int(D.afm_pass.sum())
            if kk < 1: continue
            for nm, sc in (("pan", D.pan.to_numpy()), ("lg", D.lg.to_numpy())):
                top = np.argsort(-sc, kind="stable")[:kk]; acc[nm].append((y[top].mean(), y[top].sum(), kk, y.sum()))
            fl = D.afm_pass.to_numpy().astype(bool); acc["afm"].append((y[fl].mean(), y[fl].sum(), kk, y.sum()))
        if acc["afm"]: rows.append({"species": sp, "tier": tier, "level": lvl, **{f"{k}_{m}": np.mean([a[i] for a in v]) for k, v in acc.items() for i, m in enumerate(("prec", "true", "called", "pos"))}})
T = pd.DataFrame(rows); T.round(4).to_csv("cand_match_afm.csv", index=False); print(len(T), "cells")
for k in ("pan", "lg", "afm"): print(k, "median precision %.3f" % T[f"{k}_prec"].median(), "| pooled called %d, true %d" % (T[f"{k}_called"].sum(), T[f"{k}_true"].sum()), "| min %.2f max %.2f" % (T[f"{k}_prec"].min(), T[f"{k}_prec"].max()))
print("PanPPI precision >= AFM in", int((T.pan_prec >= T.afm_prec).sum()), "of", len(T), "| LoGoPPI >= AFM in", int((T.lg_prec >= T.afm_prec).sum()), "| PanPPI > LoGoPPI in", int((T.pan_prec > T.lg_prec).sum()))
