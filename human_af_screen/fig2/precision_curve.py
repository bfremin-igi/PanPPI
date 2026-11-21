"""EXPLORATORY: precision vs true pairs found for PanPPI, LoGoPPI and ProteomeLM at several false-positive-rate thresholds (0.1%, 0.5%, 1%, 2%, 5% of each replicate's negatives), pooled over the 11 kept s700 cells x 3 replicates (same sets as precision_ci.py); AFM high-confidence call for reference. True pairs and calls are per replicate (pooled sum / 3)."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/bern_univ/"; S = ("s42", "s7", "s141")
CR = pd.read_csv(H + "cand_results.csv"); kept = {(r.species, r.tier, r.level) for r in CR[(CR.variant == "A") & CR.kept & (CR.level != "L1") & (CR.tier == "s700") & CR.species.isin(["human", "ecoli", "yeast", "fly", "mouse", "worm"])].itertuples()}   # focus species only (M. tuberculosis, P. aeruginosa excluded)
FPR = (0.001, 0.005, 0.01, 0.02, 0.05); tot = {(t, f): [0, 0] for t in ("pan", "lg", "pm") for f in FPR}; afm = [0, 0]
for sp in ["human", "ecoli", "yeast", "fly", "mouse", "worm"]:
    B = pd.read_csv(H + f"out3/{sp}/cand_scores.tsv.gz", sep="\t"); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0); PM = pd.read_csv(H + f"out3/{sp}/pm_scores.tsv.gz", sep="\t")
    B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left")
    for (tier, lvl), _ in B[B.variant == "A"].groupby(["tier", "level"]):
        if (sp, tier, lvl) not in kept: continue
        G = B[(B.variant == "A") & (B.tier == tier) & (B.level == lvl)]; P = G[G.label == 1]
        for r in sorted(set(G[G.label == 0].rep)):
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); D = pd.concat([P.sample(k, random_state=r + 1) if len(P) > k else P, N.sample(k, random_state=r + 7) if len(N) > k else N]); y = D.label.to_numpy(); fl = D.afm_pass.to_numpy().astype(bool)
            afm[0] += int(fl.sum()); afm[1] += int(y[fl].sum())
            for t in ("pan", "lg", "pm"):
                sc = D[t].to_numpy()
                for f in FPR: th = np.quantile(sc[y == 0], 1 - f); call = sc > th; tot[(t, f)][0] += int(call.sum()); tot[(t, f)][1] += int(y[call].sum())
print("AFM call: calls/replicate %.0f true/replicate %.0f precision %.3f" % (afm[0] / 3, afm[1] / 3, afm[1] / afm[0]))
rows = []
for t, nm in (("pan", "PanPPI"), ("lg", "LoGoPPI"), ("pm", "ProteomeLM-PPI")):
    for f in FPR: c, tr = tot[(t, f)]; rows.append(dict(tool=nm, fpr=f, calls_per_rep=c / 3, true_per_rep=tr / 3, precision=tr / c))
R = pd.DataFrame(rows); R.to_csv("precision_curve.csv", index=False); print(R.round(3).to_string(index=False))
