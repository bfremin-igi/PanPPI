import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
SC = "/private/tmp/claude-1618750309/-Users-<user>-oliva-hpc/a067ef54-32be-4e05-a495-9d04f41d958f/scratchpad/bern/"; SEEDS = ("s42", "s7", "s141")
SPS = sys.argv[1:] or ["human", "ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"]; rows = []
for sp in SPS:
    d = f"out2/{sp}"
    if not os.path.exists(f"{d}/clusters_id40_cluster.tsv"): print("no cluster file for", sp); continue
    ids = [l.strip() for l in open(SC + "human_ids.txt" if sp == "human" else f"../univ/{sp}/ids.txt") if l.strip()]; cl = pd.read_csv(f"{d}/clusters_id40_cluster.tsv", sep="\t", header=None, names=["rep", "mem"], dtype=str)
    reps = set(cl.rep); isrep = np.array([p in reps for p in ids]); print(f"{sp}: {len(ids)} proteins -> {len(reps)} cluster representatives ({100 * len(reps) / len(ids):.0f}%)")
    B = pd.read_csv(f"{d}/balanced_scores.tsv.gz", sep="\t"); B = B[isrep[B.i.to_numpy()] & isrep[B.j.to_numpy()]]
    L = pd.read_csv(f"{d}/logoppi_detailed.tsv.gz", sep="\t", usecols=["i", "j", "global_score", "maxsim_score", "score"]).rename(columns={"global_score": "lgG", "maxsim_score": "lgM", "score": "lgB"}); B = B.merge(L, on=["i", "j"], how="left")
    B = B.merge(pd.read_csv(f"{d}/pm_scores.tsv.gz", sep="\t").rename(columns={"pm": "pmv"}), on=["i", "j"], how="left") if os.path.exists(f"{d}/pm_scores.tsv.gz") else B.assign(pmv=np.nan)
    for s in SEEDS: B[f"bl_{s}"] = 0.5 * (B[f"g_{s}"] + B[f"m_{s}"])
    for (tier, lvl), G in B.groupby(["tier", "level"]):
        if lvl == "L1": continue
        P = G[G.label == 1]; reps_ = sorted(set(G[G.label == 0].rep)); res = {}; dg = []; nn = []
        if len(P) < 200: continue
        for r in reps_:
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); nn.append(len(N))
            Ps = P.sample(k, random_state=r + 1) if len(P) > k else P; D = pd.concat([Ps, N.sample(k, random_state=r + 7) if len(N) > k else N]); y = D.label.to_numpy()
            c = pd.concat([D.i, D.j]).value_counts(); dg.append(AU(y, np.log1p(D.i.map(c).to_numpy()) + np.log1p(D.j.map(c).to_numpy())))
            T = {"PanPPI blend": D[[f"bl_{s}" for s in SEEDS]].mean(axis=1), "PanPPI global": D[[f"g_{s}" for s in SEEDS]].mean(axis=1), "LoGoPPI blend": D.lgB, "LoGoPPI global": D.lgG, "LoGoPPI MaxSim": D.lgM}
            if D.pmv.notna().all(): T["ProteomeLM"] = D.pmv
            for nm, v in T.items(): res.setdefault(nm, []).append(AP(y, v))
        out = {"species": sp, "tier": tier, "level": lvl, "n_pos": len(P), "neg_frac": round(min(nn) / len(P), 2), "deg_only_auroc": round(float(np.mean(dg)), 3)}
        out.update({k + " aupr": float(np.mean(v)) for k, v in res.items()}); rows.append(out)
R = pd.DataFrame(rows); R["kept"] = (R.neg_frac >= 0.8) & (R.deg_only_auroc <= 0.60); R.round(4).to_csv("fig2_results_clustered.csv", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(3).to_string(index=False)); K = R[R.kept]
print("\nkept after clustering:", len(K), "cells; PanPPI blend > LoGoPPI blend in", int((K["PanPPI blend aupr"] > K["LoGoPPI blend aupr"]).sum()), "; > ProteomeLM in", int((K["PanPPI blend aupr"] > K["ProteomeLM aupr"]).sum()), "of", int(K["ProteomeLM aupr"].notna().sum()))
