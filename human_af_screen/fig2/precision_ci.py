"""Fig 2c intervals: precision of calls against STRING truth, POOLED over the kept s700 cells (L0/L2, variant A) and the three negative replicates (counts summed, as printed under the bars), with a paired protein-clustered bootstrap
(within each set, pairs are assigned to their higher-degree protein; proteins resampled with replacement, 300 draws, all tools on the identical resample; AFM-flag count, matched top-k and the 95th-percentile-of-negatives thresholds are
recomputed on every resample). Sets are built exactly as in bern_univ/eval_cand.py / eval_cand_match_afm.py. Output fig2/precision_ci.csv."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/bern_univ/"; O = "<ROOT>/human_af_screen/fig2/"; S = ("s42", "s7", "s141"); rng = np.random.default_rng(3); BDRAW = 300
CR = pd.read_csv(H + "cand_results.csv"); kept = {(r.species, r.tier, r.level) for r in CR[(CR.variant == "A") & CR.kept & (CR.level != "L1") & (CR.tier == "s700") & CR.species.isin(["human", "ecoli", "yeast", "fly", "mouse", "worm"])].itertuples()}   # focus species only (M. tuberculosis, P. aeruginosa excluded)
SETS = []
for sp in ["human", "ecoli", "yeast", "fly", "mouse", "worm"]:
    B = pd.read_csv(H + f"out3/{sp}/cand_scores.tsv.gz", sep="\t"); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0); PM = pd.read_csv(H + f"out3/{sp}/pm_scores.tsv.gz", sep="\t"); B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left"); assert B.pm.notna().all(), sp
    for (tier, lvl), _ in B[B.variant == "A"].groupby(["tier", "level"]):
        if (sp, tier, lvl) not in kept: continue
        G = B[(B.variant == "A") & (B.tier == tier) & (B.level == lvl)]; P = G[G.label == 1]
        for r in sorted(set(G[G.label == 0].rep)):
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); D = pd.concat([P.sample(k, random_state=r + 1) if len(P) > k else P, N.sample(k, random_state=r + 7) if len(N) > k else N]).reset_index(drop=True)
            c = pd.concat([D.i, D.j]).value_counts(); grp = np.where(D.i.map(c) >= D.j.map(c), D.i, D.j); u, inv = np.unique(grp, return_inverse=True)
            order = np.argsort(inv, kind="stable"); bounds = np.r_[0, np.cumsum(np.bincount(inv))]; groups = [order[bounds[q]:bounds[q + 1]] for q in range(len(u))]
            SETS.append(dict(y=D.label.to_numpy(), pan=D.pan.to_numpy(), lg=D.lg.to_numpy(), pm=D.pm.to_numpy(), afm=D.afm_pass.to_numpy().astype(bool), groups=groups, nu=len(u)))
print(len(SETS), "set x replicate tables")
def counts(Dd, ix):
    y, pan, lg, pm, afm = Dd["y"][ix], Dd["pan"][ix], Dd["lg"][ix], Dd["pm"][ix], Dd["afm"][ix]; kk = int(afm.sum()); out = {"afm": (kk, int(y[afm].sum()))}
    for nm, sc in (("pan", pan), ("lg", lg), ("pm", pm)):
        top = np.argsort(-sc, kind="stable")[:kk]; out[nm + "_match"] = (kk, int(y[top].sum()))
        thr = np.quantile(sc[y == 0], 0.95); call = sc > thr; out[nm + "_top5"] = (int(call.sum()), int(y[call].sum()))
    return out
def pooled(draw):
    tot = {}
    for Dd in SETS:
        ix = np.arange(len(Dd["y"])) if not draw else np.concatenate([Dd["groups"][q] for q in rng.integers(0, Dd["nu"], Dd["nu"])])
        for k, (c, t) in counts(Dd, ix).items(): a = tot.setdefault(k, [0, 0]); a[0] += c; a[1] += t
    return {k: (v[1] / v[0] if v[0] else np.nan, v[0], v[1]) for k, v in tot.items()}
base = pooled(False); bs = [pooled(True) for _ in range(BDRAW)]
rows = []
for k in base:
    v = np.array([b[k][0] for b in bs]); rows.append(dict(item=k, precision=base[k][0], lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5), called=base[k][1], true=base[k][2]))
for k in ("pan_match", "lg_match", "pm_match"):
    d = np.array([b[k][0] - b["afm"][0] for b in bs]); rows.append(dict(item=k + "_minus_afm", precision=base[k][0] - base["afm"][0], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
d = np.array([b["pan_match"][0] - b["lg_match"][0] for b in bs]); rows.append(dict(item="pan_minus_lg_match", precision=base["pan_match"][0] - base["lg_match"][0], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
d = np.array([b["pan_match"][0] - b["pm_match"][0] for b in bs]); rows.append(dict(item="pan_minus_pm_match", precision=base["pan_match"][0] - base["pm_match"][0], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
R = pd.DataFrame(rows); R.round(4).to_csv(O + "precision_ci.csv", index=False); print(R.round(3).to_string(index=False))
