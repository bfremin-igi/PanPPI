"""Paired protein-clustered bootstrap for Fig. 1 b, c, d (test-set noise only; training-seed noise is reported separately as the seed range).
Resampling unit = protein: each pair is assigned to its higher-degree endpoint (degree = number of pairs in the file), proteins are resampled with replacement, and EVERY score vector is evaluated on the same
resample (paired). PanPPI = mean AUPR over its 3 seeds, TUnA (panel b) = mean over its 3 released seeds, other tools = one model. Reports full-data AUPR (checked against the figure tables), mean paired
difference PanPPI minus tool and its 95% percentile interval, and the share of draws with a positive difference. B = 1000. Inputs: boot_inputs/ (pulled from cluster 2, md5-verified), ../human_af_screen/pm/results,
../human_af_screen/xsp_bern. Output: bootstrap_fig1.csv. Run: python3 -I bootstrap_fig1.py [b|c|d ...]"""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
HERE = os.path.dirname(os.path.abspath(__file__)); BI = HERE + "/boot_inputs/"; HS = HERE + "/../human_af_screen/"; B = 1000
key = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a)
class Boot:
    def __init__(self, a, b, y, seed=0):
        deg = pd.Series(np.r_[a, b]).value_counts(); da = deg.reindex(a).to_numpy(); db = deg.reindex(b).to_numpy(); g = np.where(da >= db, a, b)
        self.ug, self.gi = np.unique(g, return_inverse=True); self.y = np.asarray(y, int); self.G = len(self.ug); self.rng = np.random.default_rng(seed); self.cache = {}
    def prep(self, name, v):
        o = np.argsort(-np.asarray(v, float), kind="stable"); self.cache[name] = (o, self.y[o], self.gi[o])
    def ap(self, name, cnt):
        o, ys, gs = self.cache[name]; w = cnt[gs]; tpw = w * ys; den = np.cumsum(w); den[den == 0] = 1; return float((np.cumsum(tpw) / den * tpw).sum() / tpw.sum())
    def auc(self, name):
        """Full-data AUROC (ties broken by order) from the cached ranking."""
        o, ys, gs = self.cache[name]; P = ys.sum(); N = len(ys) - P; neg_below = N - np.cumsum(1 - ys); return float((ys * neg_below).sum() / (P * N))
    def run(self, groups, target):
        """groups: {label: [score names]} (AUPR averaged within a label); returns full values and per-draw diffs target - other."""
        full = {k: np.mean([self.ap(n, np.ones(self.G)) for n in v]) for k, v in groups.items()}
        roc = {k: np.mean([self.auc(n) for n in v]) for k, v in groups.items()}
        draws = {k: [] for k in groups}
        for _ in range(B):
            cnt = self.rng.multinomial(self.G, np.ones(self.G) / self.G); c = cnt.astype(float)
            for k, v in groups.items(): draws[k].append(np.mean([self.ap(n, c) for n in v]))
        out = []
        for k in groups:
            if k == target: continue
            d = np.array(draws[target]) - np.array(draws[k]); out.append(dict(tool=k, aupr_tool=full[k], aupr_panppi=full[target], auroc_tool=roc[k], auroc_panppi=roc[target], diff=full[target] - full[k], boot_mean_diff=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), frac_positive=float((d > 0).mean())))
        return full, out
def reindex(df, ka, kb, col, base):
    k = key(df[ka].astype(str).to_numpy(), df[kb].astype(str).to_numpy()); s = pd.Series(df[col].to_numpy(), index=k); s = s[~s.index.duplicated()]; return s.reindex(base).to_numpy()
rows = []; which = sys.argv[1:] or ["b", "c", "d"]
def finish(panel, sp, n, npos, out):
    for o in out: rows.append(dict(panel=panel, set=sp, n_pairs=n, n_pos=npos, **o))
    print(panel, sp, "n=%d pos=%d" % (n, npos), " | ".join("%s %+.3f [%+.3f, %+.3f]" % (o["tool"], o["diff"], o["lo"], o["hi"]) for o in out), flush=True)
if "b" in which:
    for sp in ("yeast", "mouse", "fly", "worm", "ecoli"):
        runs = {"s42": "e3a_nounc_650M_800_s42_r2", "s7": "e3a_nounc_650M_800_s7", "s141": "e3a_nounc_650M_800_s141_clean2"}; P = {}
        for s, r in runs.items():
            d = pd.read_csv(BI + f"runs_ft/{r}/scores_{sp}.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); d = d.drop_duplicates("k").set_index("k"); P[s] = d
        base = P["s42"].index; V = {f"PanPPI {s}": 0.5 * (P[s].cal_global.reindex(base) + P[s].cal_maxsim.reindex(base)).to_numpy() for s in P}; y = P["s42"].label.to_numpy()
        for tool, f in (("LoGoPPI", "logoppi"), ("PPLM-PPI", "pplm"), ("PLM-interact", "plminteract"), ("TUnA s47", "tuna_s47"), ("TUnA s94", "tuna_s94"), ("TUnA s141", "tuna_s141")):
            c = pd.read_csv(BI + f"analysis_out/xsp_bench/{f}/{sp}.csv", dtype={"query": str, "text": str}); V[tool] = reindex(c, "query", "text", "score", base)
        c = pd.read_csv(HS + f"pm/results/{sp}.tsv", sep="\t", dtype={"query": str, "text": str}); V["ProteomeLM-PPI"] = reindex(c, "query", "text", "supervised_score", base)
        ok = np.all([np.isfinite(v) for v in V.values()], axis=0); print(sp, "rows", len(base), "common", int(ok.sum()), "positives", int(y[ok].sum()), flush=True)
        a = np.array([k.split("|")[0] for k in base[ok]]); b = np.array([k.split("|")[1] for k in base[ok]]); bt = Boot(a, b, y[ok])
        for n, v in V.items(): bt.prep(n, v[ok])
        groups = {"PanPPI": ["PanPPI s42", "PanPPI s7", "PanPPI s141"], "LoGoPPI": ["LoGoPPI"], "PPLM-PPI": ["PPLM-PPI"], "PLM-interact": ["PLM-interact"], "ProteomeLM-PPI": ["ProteomeLM-PPI"], "TUnA": ["TUnA s47", "TUnA s94", "TUnA s141"]}
        full, out = bt.run(groups, "PanPPI"); finish("b", sp, int(ok.sum()), int(y[ok].sum()), out)
if "c" in which:
    P = {}
    for s in ("42", "7", "141"):
        d = pd.read_csv(BI + f"runs_ft/e3a_bernett_650M_800_s{s}_h1/scores_bernett.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); P[s] = d.drop_duplicates("k").set_index("k")
    base = P["42"].index; V = {f"PanPPI s{s}": 0.5 * (P[s].cal_global.reindex(base) + P[s].cal_maxsim.reindex(base)).to_numpy() for s in P}; y = P["42"].label.to_numpy()
    for tool, f in (("LoGoPPI-Bernett", "logoppi"), ("PLM-interact", "plminteract"), ("TUnA", "tuna")):
        c = pd.read_csv(BI + f"analysis_out/bernett/{f}_bernett_test.csv", dtype={"query": str, "text": str}); V[tool] = reindex(c, "query", "text", "score", base)
    ok = np.all([np.isfinite(v) for v in V.values()], axis=0); print("bernett rows", len(base), "common", int(ok.sum()), "positives", int(y[ok].sum()), flush=True)
    a = np.array([k.split("|")[0] for k in base[ok]]); b = np.array([k.split("|")[1] for k in base[ok]]); bt = Boot(a, b, y[ok])
    for n, v in V.items(): bt.prep(n, v[ok])
    groups = {"PanPPI": ["PanPPI s42", "PanPPI s7", "PanPPI s141"], "LoGoPPI-Bernett": ["LoGoPPI-Bernett"], "PLM-interact": ["PLM-interact"], "TUnA": ["TUnA"]}
    full, out = bt.run(groups, "PanPPI"); print("full-data AUPR:", {k: round(v, 4) for k, v in full.items()}); finish("c", "bernett", int(ok.sum()), int(y[ok].sum()), out)
if "d" in which:
    for sp in ("yeast", "mouse", "fly", "worm", "ecoli"):
        Pr = pd.read_csv(HS + f"xsp_bern/out/{sp}_balanced_pairs.tsv.gz", sep="\t", dtype={"a": str, "b": str}); Pr["k"] = key(Pr.a.to_numpy(), Pr.b.to_numpy()); reps = sorted(set(Pr[Pr.label == 0].rep)); pos = Pr[Pr.label == 1]
        S = {}
        for s in ("7", "141", "42"):
            c = pd.read_csv(HS + f"xsp_bern/res/panppi_s{s}/components_{sp}.tsv", sep="\t", dtype={"a": str, "b": str}); c["k"] = key(c.a.to_numpy(), c.b.to_numpy()); S[f"PanPPI s{s}"] = c.drop_duplicates("k").set_index("k").cal_blend
        for tool, f in (("LoGoPPI-B", "logoppi"), ("PLM-interact-B", "plminteract"), ("TUnA-B", "tuna")):
            c = pd.read_csv(HS + f"xsp_bern/res/{f}/{sp}.csv", dtype={"query": str, "text": str}); c["k"] = key(c["query"].to_numpy(), c["text"].to_numpy()); S[tool] = c.drop_duplicates("k").set_index("k").score
        r = reps[0]; D = pd.concat([pos, Pr[(Pr.label == 0) & (Pr.rep == r)]]); V = {n: s.reindex(D.k).to_numpy() for n, s in S.items()}; y = D.label.to_numpy(); ok = np.all([np.isfinite(v) for v in V.values()], axis=0)
        a = np.array([k.split("|")[0] for k in D.k[ok]]); b = np.array([k.split("|")[1] for k in D.k[ok]]); bt = Boot(a, b, y[ok])
        for n, v in V.items(): bt.prep(n, v[ok])
        groups = {"PanPPI": ["PanPPI s7", "PanPPI s141", "PanPPI s42"], "LoGoPPI-B": ["LoGoPPI-B"], "PLM-interact-B": ["PLM-interact-B"], "TUnA-B": ["TUnA-B"]}
        full, out = bt.run(groups, "PanPPI"); finish("d", f"{sp} (replicate {r})", int(ok.sum()), int(y[ok].sum()), out)
R = pd.DataFrame(rows); R.round(4).to_csv(HERE + "/bootstrap_fig1_" + "".join(which) + ".csv", index=False); print("wrote bootstrap_fig1_" + "".join(which) + ".csv")
