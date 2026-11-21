"""Supplementary Fig. 6: AUPR of the Bernett-trained models on random size-matched negatives and on hard negatives that share a GO cellular component (a) or a Pfam domain (b), positives drawn from positives that also share (LoGoPPI's design).
PanPPI = mean of three seeds. Whiskers: 2.5-97.5 percentiles of 2,000 bootstrap replicates. Reads hard_negatives_go_pfam.csv."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("hard_negatives_go_pfam.csv"); T = ["PanPPI", "LoGoPPI-Bernett", "PLM-interact-Bernett", "TUnA-Bernett"]; NM = {"PanPPI": "PanPPI", "LoGoPPI-Bernett": "LoGoPPI", "PLM-interact-Bernett": "PLM-interact", "TUnA-Bernett": "TUnA"}
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
for a, (an, ttl) in zip(axs, (("GO cellular component", "a  Negatives share a GO cellular component"), ("Pfam domain", "b  Negatives share a Pfam domain"))):
    d = R[R.annotation == an].set_index("tool"); n = int(d.hard_negatives.iloc[0]); x = np.arange(len(T)); w = 0.36
    for j, (col, lab, c) in enumerate((("aupr_random", "Random negatives", "#B9BEC4"), ("aupr_hard", "Hard negatives", "#4A3FA0"))):
        vals = np.array([d.loc[t, col] for t in T]); lo = np.array([d.loc[t, col + "_lo"] for t in T]); hi = np.array([d.loc[t, col + "_hi"] for t in T])
        a.bar(x + (j - 0.5) * w, vals, w, color=c, label=lab); a.errorbar(x + (j - 0.5) * w, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", elinewidth=1, capsize=2)
        for xi, vi, hh in zip(x + (j - 0.5) * w, vals, hi): a.text(xi, hh + 0.008, f"{vi:.3f}".lstrip("0"), ha="center", fontsize=8, fontweight="bold" if (j == 1 and False) else "normal")
    a.set_xticks(x); a.set_xticklabels([NM[t] for t in T], fontsize=9); a.set_ylim(0.5, 1.0); a.set_title(f"{ttl} (n = {n:,} per class)", loc="left", fontsize=9); a.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("AUPR (1:1 prevalence)"); axs[0].legend(frameon=False, fontsize=8.5, loc="upper right"); plt.tight_layout(); plt.savefig("figS_hardneg.png", dpi=190); plt.savefig("figS_hardneg.svg"); print("ok")
