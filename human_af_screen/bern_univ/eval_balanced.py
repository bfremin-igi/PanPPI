import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
SPS = sys.argv[1:] or ["ecoli", "mtb", "paer", "yeast"]; SEEDS = ("s42", "s7", "s141"); rng = np.random.default_rng(0); rows = []
def tools(B):
    T = {}
    for s in SEEDS: T[f"blend_{s}"] = 0.5 * (B[f"g_{s}"] + B[f"m_{s}"]); T[f"global_{s}"] = B[f"g_{s}"]
    T["lg"] = B["lg"]; return T
for sp in SPS:
    B = pd.read_csv(f"out/{sp}/balanced_scores.tsv.gz", sep="\t"); meta = {(m["tier"], m["level"]): m for m in json.load(open(f"out/{sp}/balanced_meta.json"))}
    for (tier, lvl), G in B.groupby(["tier", "level"], sort=False):
        P = G[G.label == 1]; reps = sorted(set(G[G.label == 0].rep)); m = meta[(tier, lvl)]; nneg = int(np.mean([len(G[(G.label == 0) & (G.rep == r)]) for r in reps])); npos = len(P)
        res = {k: [] for k in ["blend", "global", "lg"]}; per_seed = {k: [] for k in ("blend", "global")}; ap_ = {k: [] for k in ["blend", "global", "lg"]}; deg = []
        for r in reps:
            N_ = G[(G.label == 0) & (G.rep == r)]; k = min(len(N_), len(P)); Ps = P.sample(k, random_state=int(r) + 1); D = pd.concat([Ps, N_.sample(k, random_state=int(r) + 7)]); y = D.label.values
            dg = pd.Series(np.r_[D.i.values, D.j.values]).value_counts(); hub = np.log1p(D.i.map(dg).values) + np.log1p(D.j.map(dg).values); deg.append(AU(y, hub))
            T = tools(D)
            bs = [AU(y, T[f"blend_{s}"]) for s in SEEDS]; gs = [AU(y, T[f"global_{s}"]) for s in SEEDS]; res["blend"].append(np.mean(bs)); res["global"].append(np.mean(gs)); res["lg"].append(AU(y, T["lg"]))
            ap_["blend"].append(np.mean([AP(y, T[f"blend_{s}"]) for s in SEEDS])); ap_["global"].append(np.mean([AP(y, T[f"global_{s}"]) for s in SEEDS])); ap_["lg"].append(AP(y, T["lg"]))
            per_seed["blend"].append(bs); per_seed["global"].append(gs)
        ps = {k: np.mean(v, 0) for k, v in per_seed.items()}
        rows.append(dict(species=sp, tier=tier, level=lvl, n_pos=npos, n_neg=nneg, neg_frac=round(nneg / npos, 2), proteins=m.get("n_proteins"), deg_only_auroc=round(float(np.mean(deg)), 3),
                         auroc_blend=round(float(np.mean(res["blend"])), 3), auroc_blend_seeds="/".join("%.3f" % x for x in ps["blend"]), auroc_global=round(float(np.mean(res["global"])), 3), auroc_lg=round(float(np.mean(res["lg"])), 3),
                         aupr_blend=round(float(np.mean(ap_["blend"])), 3), aupr_global=round(float(np.mean(ap_["global"])), 3), aupr_lg=round(float(np.mean(ap_["lg"])), 3)))
D = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(D.to_string(index=False)); D.to_csv("balanced_eval_" + "_".join(SPS) + ".csv", index=False)
