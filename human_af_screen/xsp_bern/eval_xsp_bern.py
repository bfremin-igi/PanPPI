import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
SETS = sys.argv[1:] or ["yeast", "mouse", "fly", "worm", "ecoli"]; rows = []
for sp in SETS:
    P = pd.read_csv(f"out/{sp}_balanced_pairs.tsv.gz", sep="\t", dtype={"a": str, "b": str}); key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a); P["k"] = key(P.a.values, P.b.values)
    S = {}
    for s in ("7", "141", "42"):
        f = f"res/panppi_s{s}/components_{sp}.tsv"
        if os.path.exists(f):
            c = pd.read_csv(f, sep="\t", dtype={"a": str, "b": str}); c["k"] = key(c.a.values, c.b.values); c = c.drop_duplicates("k").set_index("k")
            S[f"PanPPI blend s{s}"] = c.cal_blend; S[f"PanPPI global s{s}"] = c.cal_global; S[f"PanPPI MaxSim s{s}"] = c.cal_maxsim
    f = f"res/logoppi/{sp}.csv"
    if os.path.exists(f):
        c = pd.read_csv(f, dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); c = c.drop_duplicates("k").set_index("k"); S["LoGoPPI-B blend"] = c.score; S["LoGoPPI-B global"] = c.global_score; S["LoGoPPI-B MaxSim"] = c.maxsim_score
    f = f"res/tuna/{sp}.csv"
    if os.path.exists(f):
        c = pd.read_csv(f, dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); S["TUnA-B"] = c.drop_duplicates("k").set_index("k").score
    f = f"res/plminteract/{sp}.csv"
    if os.path.exists(f):
        c = pd.read_csv(f, dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); S["PLM-interact-B"] = c.drop_duplicates("k").set_index("k").score
    pos = P[P.label == 1]; reps = sorted(set(P[P.label == 0].rep))
    for name, ser in S.items():
        au, ap = [], []
        for r in reps:
            D = pd.concat([pos, P[(P.label == 0) & (P.rep == r)]]); v = ser.reindex(D.k).to_numpy(); ok = ~np.isnan(v); y = D.label.to_numpy()[ok]; au.append(AU(y, v[ok])); ap.append(AP(y, v[ok]))
        rows.append((sp, name, np.mean(au), np.mean(ap), len(reps)))
D = pd.DataFrame(rows, columns=["set", "tool", "auroc", "aupr", "reps"]); pd.set_option("display.width", 200)
T = D.pivot(index="tool", columns="set", values="auroc").round(3); print("AUROC (balanced 1:1, mean of the 2 negative replicates)"); print(T.to_string())
T = D.pivot(index="tool", columns="set", values="aupr").round(3); print("\nAUPR (chance .50)"); print(T.to_string())
D.to_csv("xsp_bern_results.csv", index=False)
