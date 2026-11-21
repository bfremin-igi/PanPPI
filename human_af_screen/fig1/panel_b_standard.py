import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
B = "/private/tmp/claude-1618750309/-Users-<user>-oliva-hpc/a067ef54-32be-4e05-a495-9d04f41d958f/scratchpad/"
RUNS = {"s42": "e3a_nounc_650M_800_s42_r2", "s7": "e3a_nounc_650M_800_s7", "s141": "e3a_nounc_650M_800_s141_clean2"}; SETS = [("yeast", "Yeast"), ("mouse", "Mouse"), ("fly", "Fly"), ("worm", "Worm"), ("ecoli", "E. coli")]
key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a); rows = []; brows = []
for sp, nm in SETS:
    u = pd.read_csv(B + f"unc/pairs_{sp}.tsv.gz", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label"]); u["k"] = key(u.a.values, u.b.values); y = u.label.to_numpy(); n = len(u)
    def add(tool, s, v, kind="score"):
        v = np.asarray(v, float); ok = np.isfinite(v)
        if ok.sum() < n * 0.99: print("WARNING", nm, tool, s, "missing", n - ok.sum()); 
        rows.append((nm, tool, s, AU(y[ok], v[ok]), AP(y[ok], v[ok]), int(ok.sum()), int(y[ok].sum())))
    for s, rn in RUNS.items():
        f = B + f"comp/{rn}/scores_{sp}.tsv"
        if not os.path.exists(f): print("missing", f); continue
        c = pd.read_csv(f, sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "cal_global", "cal_maxsim"]); c["k"] = key(c.a.values, c.b.values); c = c.drop_duplicates("k").set_index("k").reindex(u.k)
        add("PanPPI blend", s, 0.5 * (c.cal_global.values + c.cal_maxsim.values)); add("PanPPI global", s, c.cal_global.values); add("PanPPI MaxSim", s, c.cal_maxsim.values)
    for tool, f in (("LoGoPPI", "logoppi"), ("PPLM-PPI", "pplm"), ("PLM-interact", "plminteract")):
        c = pd.read_csv(B + f"unc/{f}_{sp}.csv", dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); add(tool, "-", c.drop_duplicates("k").set_index("k").score.reindex(u.k).values)
    f = f"<ROOT>/human_af_screen/pm/results/{sp}.tsv"
    if os.path.exists(f): c = pd.read_csv(f, sep="\t", dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); add("ProteomeLM-PPI", "-", c.drop_duplicates("k").set_index("k").supervised_score.reindex(u.k).values)
    for s in ("47", "94", "141"):
        f = B + f"unc/tuna_s{s}_{sp}.csv"
        if os.path.exists(f): c = pd.read_csv(f, dtype={"query": str, "text": str}); c["k"] = key(c["query"].values, c["text"].values); add("TUnA", "s" + s, c.drop_duplicates("k").set_index("k").score.reindex(u.k).values)
R = pd.DataFrame(rows, columns=["set", "tool", "seed", "auroc", "aupr", "n_scored", "n_pos"]); R.round(4).to_csv("panel_b_standard_per_seed.csv", index=False)
S = R.groupby(["set", "tool"]).agg(auroc=("auroc", "mean"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), aupr=("aupr", "mean"), aupr_min=("aupr", "min"), aupr_max=("aupr", "max"), n_seeds=("seed", "count"), n_pos=("n_pos", "first")).reset_index(); S.round(4).to_csv("panel_b_standard_summary.csv", index=False)
order = ["PanPPI blend", "PanPPI global", "PanPPI MaxSim", "LoGoPPI", "PPLM-PPI", "PLM-interact", "TUnA", "ProteomeLM-PPI"]; sets = [n for _, n in SETS]
for met in ("aupr", "auroc"):
    P = S.pivot(index="tool", columns="set", values=met).loc[order][sets]; print(f"\nSTANDARD cross-species, {met.upper()} (PanPPI and TUnA: mean over 3 seeds; full test files, original 1:10)"); print(P.round(3).to_string())
print("\nPanPPI blend per seed AUPR:"); print(R[R.tool == "PanPPI blend"].pivot(index="seed", columns="set", values="aupr")[sets].round(3).to_string())
