"""Supplementary Fig. 7: retention of AFM-confident pairs by STRING support. Two panels (top 1% and top 10% of all pairs), rows = cells, three STRING tiers per row (<400, 400-699, >=700), PanPPI filled / LoGoPPI open, 95% protein-clustered
bootstrap intervals, n per tier. Dashed line = retention expected from random selection. Reads retention_by_string.csv."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("retention_by_string.csv"); SETS = ["Human (new pairs)", "Yeast", "Mouse", "Fly", "Worm", "E. coli"]; TIERS = [("<400", "#BFC5CC"), ("400-699", "#8E96A3"), (">=700", "#3D4651")]; PAN, LG = "#4A3FA0", "#E0566A"
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
for a, (f, ttl) in zip(axs, ((1e-2, "a  Top 1% of all pairs"), (1e-1, "b  Top 10% of all pairs"))):
    for r, sp in enumerate(SETS):
        for k, (t, shade) in enumerate(TIERS):
            d = R[(R.set == sp) & (R.string_tier == t) & np.isclose(R.fraction, f)]
            if d.empty: continue
            d = d.iloc[0]; y = r + (k - 1) * 0.24
            a.plot([d.PanPPI_lo, d.PanPPI_hi], [y - 0.045, y - 0.045], color=PAN, lw=1.6); a.plot([d.LoGoPPI_lo, d.LoGoPPI_hi], [y + 0.045, y + 0.045], color=LG, lw=1.6)
            a.scatter([d.PanPPI], [y - 0.045], s=34, color=PAN, zorder=3, edgecolor="white", linewidth=0.6); a.scatter([d.LoGoPPI], [y + 0.045], s=34, facecolor="white", edgecolor=LG, linewidth=1.5, zorder=3)
            a.text(1.015, y, f"{t}  n={int(d.n_afm_confident):,}", fontsize=7, va="center", color="#555555", transform=a.get_yaxis_transform()) if f == 1e-1 else None
    a.axvline(f, color="#999999", ls="--", lw=1); a.set_yticks(range(len(SETS))); a.set_yticklabels(SETS, fontsize=9); a.set_ylim(len(SETS) - 0.45, -0.55); a.set_xlim(0, 1); a.set_xlabel("Fraction of AFM-confident pairs kept", fontsize=9); a.set_title(ttl, loc="left", fontsize=9.5, fontweight="bold"); a.spines[["top", "right"]].set_visible(False); a.grid(axis="x", color="#EEEEEE", lw=0.8); a.set_axisbelow(True)
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color=PAN, lw=0, markersize=6, label="PanPPI"), Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=LG, color=LG, lw=0, markersize=6, label="LoGoPPI"), Line2D([0], [0], color="#999999", ls="--", label="random selection")]
axs[0].legend(handles=leg, frameon=False, fontsize=8.5, loc="lower right"); fig.text(0.5, 0.005, "Within each cell, rows are STRING physical score < 400, 400-699 and >= 700 (top to bottom); n = AFM-confident pairs in the tier.", ha="center", fontsize=8, color="#555555")
plt.tight_layout(rect=(0, 0.03, 0.9, 1)); plt.savefig("supp_fig7_string_tiers.png", dpi=190); plt.savefig("supp_fig7_string_tiers.svg"); print("ok")
