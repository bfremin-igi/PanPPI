"""Uncertainty boost on the standard cross-species files (and, with --bernett later, on Bernett). PROTOCOL PRE-REGISTERED in paper/supplement_plan.md before any Bernett number exists.
Features per pair: z = calibrated blend (cal_blend); logvar = log post-hoc GP variance; protein distance = mean cosine distance of each protein's pooled projected embedding to its 10 nearest TRAINING proteins,
pair features dmin, dmax; MaxSim evidence = mean gap, |gap_a - gap_b|, mean sd, |sa - sb|. Boosts: B0 z | B1 z+logvar | B2 z+logvar+dist | B3 z+logvar+dist+evidence ("trio"); logistic regression, standardised, L2 C=1.
Standard files: leave-one-species-out over the 8 original species (fit on the other 7 pooled), evaluate on the held-out species (core five reported). Unmatched evaluation only. Per seed; AUPR/AUROC; paired protein-clustered bootstrap (B3 - B0)."""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
HERE = os.path.dirname(os.path.abspath(__file__)); UI = HERE + "/unc_inputs/"; key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)
RUNS = {"42": "e3a_nounc_650M_800_s42_r2", "7": "e3a_nounc_650M_800_s7", "141": "e3a_nounc_650M_800_s141_clean2"}; SP8 = ["yeast", "mouse", "fly", "worm", "ecoli"]; CORE = SP8   # core five only: the three fitting-only bacterial/archaeal sets (208964, 224308, 83332) are no longer used
def load_emb(prefix):
    return np.load(prefix + ".npy").astype(np.float32), [l.strip() for l in open(prefix + ".ids") if l.strip()]
def knn_dist(E, ids, T, k=10, chunk=2000):
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9); Tn = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9); out = np.zeros(len(En), np.float32)
    for i in range(0, len(En), chunk):
        S = En[i:i + chunk] @ Tn.T; top = np.partition(S, -k, axis=1)[:, -k:]; out[i:i + chunk] = 1.0 - top.mean(1)
    return dict(zip(ids, out))
def build(posthoc_tsv, uncert_pairs, emb_prefix, train):
    g = pd.read_csv(posthoc_tsv, sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_blend", "var"]); u = pd.read_csv(uncert_pairs, sep="\t", dtype={"a": str, "b": str})
    g["k"] = key(g.a.to_numpy(), g.b.to_numpy()); u["k"] = key(u.a.to_numpy(), u.b.to_numpy()); g = g.drop_duplicates("k"); u = u.drop_duplicates("k")
    d = g.merge(u[["k", "label", "sa", "sb", "gap_a", "gap_b", "sd_a", "sd_b"]], on="k", suffixes=("", "_u")); assert (d.label == d.label_u).all(), "label mismatch"
    E, ids = load_emb(emb_prefix); dm = knn_dist(E, ids, train); da = d.a.map(dm).to_numpy(); db = d.b.map(dm).to_numpy()
    F = pd.DataFrame({"z": d.cal_blend.to_numpy(), "logvar": np.log(np.maximum(d["var"].to_numpy(), 1e-12)), "dmin": np.minimum(da, db), "dmax": np.maximum(da, db),
                      "gap_mean": (d.gap_a + d.gap_b).to_numpy() / 2, "gap_asym": np.abs(d.gap_a - d.gap_b).to_numpy(), "sd_mean": (d.sd_a + d.sd_b).to_numpy() / 2, "s_asym": np.abs(d.sa - d.sb).to_numpy()})
    return F, d.label.to_numpy().astype(int), d.a.to_numpy(), d.b.to_numpy()
SETS = {"B0": ["z"], "B1": ["z", "logvar"], "B2": ["z", "logvar", "dmin", "dmax"], "B3": ["z", "logvar", "dmin", "dmax", "gap_mean", "gap_asym", "sd_mean", "s_asym"]}
def fit_predict(Ftr, ytr, Fte, cols):
    if cols == ["z"]: return Fte["z"].to_numpy()
    sc = StandardScaler().fit(Ftr[cols]); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Ftr[cols]), ytr); return m.decision_function(sc.transform(Fte[cols]))
if __name__ == "__main__":
    seeds = sys.argv[1:] or ["42", "141"]; rows = []
    for s in seeds:
        UD = UI + f"analysis_out/uncert_s{s}/"; train, _ = load_emb(UD + "prot_humantrain"); D = {}
        for sp in SP8:
            F, y, a, b = build(UI + f"runs_ft/{RUNS[s]}/posthoc_gp/scores_{sp}_posthoc.tsv", UD + f"pairs_{sp}.tsv.gz", UD + f"prot_{sp}", train); D[sp] = (F, y, a, b); print("seed", s, sp, len(y), "pos", int(y.sum()), flush=True)
        for te in SP8:
            Ftr = pd.concat([D[x][0] for x in SP8 if x != te]); ytr = np.concatenate([D[x][1] for x in SP8 if x != te]); Fte, yte = D[te][0], D[te][1]
            for name, cols in SETS.items():
                sc = fit_predict(Ftr, ytr, Fte, cols); rows.append(dict(seed=s, set=te, boost=name, aupr=AP(yte, sc), auroc=AU(yte, sc), n=len(yte), pos=int(yte.sum())))
    R = pd.DataFrame(rows); R.round(5).to_csv(os.path.join(HERE, "unc_boost_standard.csv"), index=False)
    M = R[R.set.isin(CORE)].groupby(["boost", "set"]).aupr.mean().unstack("set")[CORE]; M["macro5"] = M.mean(1); print("\nAUPR, mean over seeds %s, held-out species (LOSO over the 8 originals), unmatched" % seeds); print(M.round(4).to_string())
    M = R[R.set.isin(CORE)].groupby(["boost", "set"]).auroc.mean().unstack("set")[CORE]; M["macro5"] = M.mean(1); print("\nAUROC"); print(M.round(4).to_string())
