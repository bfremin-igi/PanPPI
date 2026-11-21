import os, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
SEEDS = ("s42", "s7", "s141"); SPS = ["human", "ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"]; FPRS = (0.01, 0.05, 0.10); rng = np.random.default_rng(0)
R0 = pd.read_csv("fig2_results_long.csv"); KEPT = {(r.species, r.tier, r.level) for r in R0.itertuples() if r.kept}; rec_rows = []; ci_rows = []
def recall_at(pos, neg, f): t = np.quantile(neg, 1 - f); return float((pos > t).mean())
for sp in SPS:
    d = f"out2/{sp}"; B = pd.read_csv(f"{d}/balanced_scores.tsv.gz", sep="\t")
    L = pd.read_csv(f"{d}/logoppi_detailed.tsv.gz", sep="\t", usecols=["i", "j", "global_score", "score"]).rename(columns={"global_score": "lgG", "score": "lgB"}); B = B.merge(L, on=["i", "j"], how="left")
    B = B.merge(pd.read_csv(f"{d}/pm_scores.tsv.gz", sep="\t").rename(columns={"pm": "pmv"}), on=["i", "j"], how="left") if os.path.exists(f"{d}/pm_scores.tsv.gz") else B.assign(pmv=np.nan)
    B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in SEEDS], axis=0)   # seed-ensemble PanPPI blend
    for (tier, lvl), G in B.groupby(["tier", "level"]):
        if (sp, tier, lvl) not in KEPT or lvl == "L1": continue
        P = G[G.label == 1]; reps = sorted(set(G[G.label == 0].rep)); tools = {"PanPPI": "pan", "LoGoPPI blend": "lgB", "LoGoPPI global": "lgG"}
        if G.pmv.notna().all(): tools["ProteomeLM"] = "pmv"
        for nm, col in tools.items():
            out = {"species": sp, "tier": tier, "level": lvl, "tool": nm, "n_pos": len(P)}
            for f in FPRS: out[f"recall@FPR{int(f * 100)}%"] = float(np.mean([recall_at(P[col].to_numpy(), G[(G.label == 0) & (G.rep == r)][col].to_numpy(), f) for r in reps]))
            rec_rows.append(out)
        # paired protein-clustered bootstrap, replicate 0, PanPPI ensemble vs LoGoPPI blend
        N = G[(G.label == 0) & (G.rep == 0)]; D = pd.concat([P, N]).reset_index(drop=True); y = D.label.to_numpy(); deg = pd.concat([D.i, D.j]).value_counts(); key = np.where(D.i.map(deg).to_numpy() >= D.j.map(deg).to_numpy(), D.i, D.j)
        u, inv = np.unique(key, return_inverse=True); res = {k: [] for k in ("pan", "lgB", "d")}
        for _ in range(200):
            w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float); keep = w > 0
            a = AP(y[keep], D.pan.to_numpy()[keep], sample_weight=w[keep]); b = AP(y[keep], D.lgB.to_numpy()[keep], sample_weight=w[keep]); res["pan"].append(a); res["lgB"].append(b); res["d"].append(a - b)
        pa, pb = AP(y, D.pan), AP(y, D.lgB); ci_rows.append({"species": sp, "tier": tier, "level": lvl, "n_pos": len(P), "n_groups": len(u), "PanPPI ens aupr": pa, "LoGoPPI blend aupr": pb, "diff": pa - pb, "diff_lo": np.percentile(res["d"], 2.5), "diff_hi": np.percentile(res["d"], 97.5), "PanPPI_lo": np.percentile(res["pan"], 2.5), "PanPPI_hi": np.percentile(res["pan"], 97.5)})
        print(sp, tier, lvl, "diff %.3f [%.3f, %.3f]" % (pa - pb, np.percentile(res["d"], 2.5), np.percentile(res["d"], 97.5)), flush=True)
pd.DataFrame(rec_rows).round(4).to_csv("fig2_recall_at_fpr.csv", index=False); C = pd.DataFrame(ci_rows); C.round(4).to_csv("fig2_ci.csv", index=False)
print("\ncells where the 95% CI of (PanPPI ensemble - LoGoPPI blend) AUPR excludes 0:", int((C.diff_lo > 0).sum()), "of", len(C), "| includes 0:", C[C.diff_lo <= 0][["species", "tier", "level", "n_pos", "diff", "diff_lo", "diff_hi"]].round(3).to_dict("records"))
