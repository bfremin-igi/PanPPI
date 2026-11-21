"""Supplementary Fig. 4 (bar version, like Fig. 1): AUPR of the calibrated blend (baseline) and of the blend with each uncertainty layer added cumulatively: + pair-level GP variance (B1), + protein distance (B2), + MaxSim evidence (B3).
a: standard cross-species files (regression fitted on the other species); b: Bernett test (fitted on the Bernett validation set; and fitted on the standard species as a control); c: Bernett-style cross-species sets (fitted on the other four).
Bars: mean of three seeds; circles: seeds. Axes start at 0.5. The control fitted on the standard species (Bernett test) is in Supplementary Table 6 only. Reads ../paper/unc_boost_standard.csv and unc_bernett.csv."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P = "../paper/"; SB = pd.read_csv(P + "unc_boost_standard.csv"); UB = pd.read_csv(P + "unc_bernett.csv")
SP = ["yeast", "mouse", "fly", "worm", "ecoli"]; NM = {"yeast": "Yeast", "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "ecoli": "E. coli"}
BO = ["B0", "B1", "B2", "B3"]; COL = {"B0": "#8A8F96", "B1": "#2A9DB5", "B2": "#E39B2D", "B3": "#4A3FA0"}
LAB = {"B0": "Calibrated blend (baseline)", "B1": "+ pair-level GP variance", "B2": "+ protein distance", "B3": "+ MaxSim evidence"}
def groups_std(): return [(NM[k], SB[SB.set == k]) for k in SP]
def groups_bern(): return [("fit: Bernett\nvalidation", UB[UB.analysis == "A_bernett_val_fit"]), ("fit: standard\nspecies", UB[UB.analysis == "B_transfer_from_standard"])]
def groups_xsp(): return [(NM[k], UB[(UB.analysis == "C_bernett_style_LOSO") & (UB.set == k)]) for k in SP] + [("Human\n(Bernett test)", UB[UB.analysis == "A_bernett_val_fit"])]
def draw(ax, groups, title, ylim):
    n = len(groups); w = 0.19
    for gi, (lab, d) in enumerate(groups):
        for bi, b in enumerate(BO):
            v = d[d.boost == b].sort_values("seed").aupr.to_numpy(); x = gi + (bi - 1.5) * w; m = v.mean()
            ax.bar(x, m, w * 0.92, color=COL[b], zorder=2); ax.plot([x] * len(v), v, "o", color="white", mec="#333333", mew=0.7, ms=2.8, zorder=4)
            ax.text(x, max(v.max(), m) + 0.006, f"{m:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=6.2, rotation=90, fontweight="bold" if b == "B3" else "normal")
    ax.set_xticks(range(n)); ax.set_xticklabels([g[0] for g in groups], fontsize=9); ax.set_ylim(*ylim); ax.set_ylabel("AUPR"); ax.set_title(title, loc="left", fontsize=9.5); ax.spines[["top", "right"]].set_visible(False)
fig = plt.figure(figsize=(10.4, 7.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.34], hspace=0.42, wspace=0.05)
a = fig.add_subplot(gs[0, 0]); c = fig.add_subplot(gs[1, 0])
draw(a, groups_std(), "a  Standard cross-species files", (0.5, 1.06)); draw(c, groups_xsp(), "b  Bernett protocol", (0.5, 0.86))
leg = fig.add_subplot(gs[:, 1]); leg.axis("off"); leg.legend([Patch(color=COL[x]) for x in BO], [LAB[x] for x in BO], frameon=False, fontsize=9, loc="center left", title="Layers added cumulatively\n(circles: seeds)", title_fontsize=9, alignment="left")
plt.savefig("figS_uncertainty.png", dpi=190, bbox_inches="tight"); plt.savefig("figS_uncertainty.svg", bbox_inches="tight"); print("ok")
