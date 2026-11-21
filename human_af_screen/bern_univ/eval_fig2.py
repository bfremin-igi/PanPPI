import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
SPS = ["human", "ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"]; SEEDS = ("s42", "s7", "s141"); rows = []; drops = []
for sp in SPS:
    d = f"out2/{sp}"; B = pd.read_csv(f"{d}/balanced_scores.tsv.gz", sep="\t"); meta = {(m["tier"], m["level"]): m for m in json.load(open(f"{d}/balanced_meta.json")) if "note" not in m}
    L = pd.read_csv(f"{d}/logoppi_detailed.tsv.gz", sep="\t", usecols=["i", "j", "global_score", "maxsim_score", "score"]).rename(columns={"global_score": "lgG", "maxsim_score": "lgM", "score": "lgB"}); B = B.merge(L, on=["i", "j"], how="left")
    if os.path.exists(f"{d}/pm_scores.tsv.gz"): B = B.merge(pd.read_csv(f"{d}/pm_scores.tsv.gz", sep="\t").rename(columns={"pm": "pmv"}), on=["i", "j"], how="left")
    else: B["pmv"] = np.nan
    assert B.lgB.notna().all(), (sp, "missing LoGoPPI rows")
    for s in SEEDS: B[f"bl_{s}"] = 0.5 * (B[f"g_{s}"] + B[f"m_{s}"])
    for (tier, lvl), m in meta.items():
        G = B[(B.tier == tier) & (B.level == lvl)]; P = G[G.label == 1]; nneg = min(m["neg_per_rep"]); frac = nneg / m["n_pos"]; reps = sorted(set(G[G.label == 0].rep))
        res = {}; dg = []
        for r in reps:
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); Ps = P.sample(k, random_state=r + 1) if len(P) > k else P; D = pd.concat([Ps, N.sample(k, random_state=r + 7) if len(N) > k else N]); y = D.label.to_numpy()
            c = pd.concat([D.i, D.j]).value_counts(); hub = np.log1p(D.i.map(c).to_numpy()) + np.log1p(D.j.map(c).to_numpy()); dg.append(AU(y, hub))
            T = {f"PanPPI blend {s}": D[f"bl_{s}"] for s in SEEDS}; T.update({f"PanPPI global {s}": D[f"g_{s}"] for s in SEEDS}); T.update({f"PanPPI MaxSim {s}": D[f"m_{s}"] for s in SEEDS}); T.update({"LoGoPPI blend": D.lgB, "LoGoPPI global": D.lgG, "LoGoPPI MaxSim": D.lgM})
            if D.pmv.notna().all(): T["ProteomeLM"] = D.pmv
            for nm, v in T.items(): res.setdefault(nm, []).append((AU(y, v), AP(y, v)))
        deg = float(np.mean(dg)); ok = frac >= 0.8 and deg <= 0.60
        if not ok: drops.append((sp, tier, lvl, m["n_pos"], round(frac, 2), round(deg, 3))); 
        out = {"species": sp, "tier": tier, "level": lvl, "n_pos": m["n_pos"], "neg_frac": round(frac, 2), "deg_only_auroc": round(deg, 3), "kept": ok}
        for tool in ("PanPPI blend", "PanPPI global", "PanPPI MaxSim"):
            a = [np.mean([x[0] for x in res[f"{tool} {s}"]]) for s in SEEDS]; p = [np.mean([x[1] for x in res[f"{tool} {s}"]]) for s in SEEDS]
            out[tool + " auroc"] = np.mean(a); out[tool + " auroc_min"] = min(a); out[tool + " auroc_max"] = max(a); out[tool + " aupr"] = np.mean(p); out[tool + " aupr_min"] = min(p); out[tool + " aupr_max"] = max(p)
        for tool in ("LoGoPPI blend", "LoGoPPI global", "LoGoPPI MaxSim", "ProteomeLM"):
            if tool in res: out[tool + " auroc"] = np.mean([x[0] for x in res[tool]]); out[tool + " aupr"] = np.mean([x[1] for x in res[tool]])
        rows.append(out)
R = pd.DataFrame(rows); R.round(4).to_csv("fig2_results_long.csv", index=False); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
K = R[R.kept & (R.level != "L1")]
for met in ("aupr", "auroc"):
    cols = ["species", "tier", "level", "n_pos", "deg_only_auroc"] + [f"{t} {met}" for t in ("PanPPI blend", "PanPPI global", "PanPPI MaxSim", "LoGoPPI blend", "LoGoPPI global", "LoGoPPI MaxSim", "ProteomeLM")]
    print(f"\n{met.upper()} on the kept cells (L0 all proteins / L2 no homolog; PanPPI = 3-seed mean)"); print(K[cols].round(3).to_string(index=False))
print("\nHuman L1 (clean strata 3-5):"); print(R[(R.level == "L1")][["species", "tier", "n_pos", "neg_frac", "deg_only_auroc", "kept", "PanPPI blend aupr", "PanPPI global aupr", "LoGoPPI blend aupr", "LoGoPPI global aupr", "ProteomeLM aupr"]].round(3).to_string(index=False))
print("\nDROPPED cells (species, tier, level, n_pos, neg_frac, degree-only AUROC):"); [print("  ", d) for d in drops]
