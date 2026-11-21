"""Shared paired protein-clustered bootstrap (see bootstrap_fig1.py for the definition)."""
import numpy as np, pandas as pd
B = 1000
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
