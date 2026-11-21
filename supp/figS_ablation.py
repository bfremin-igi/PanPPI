"""Supplementary Fig. 1 (redone): PanPPI (first) vs a control trained without the ranking loss; AUPR on the x axis for the blend and each branch, five D-SCRIPT species. Reads ablation_ctrl.csv (from ablation_ctrl.py).
Dots: mean of three seeds (whiskers: seed range); right: PanPPI minus control with the 95% paired protein-clustered bootstrap interval."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("ablation_ctrl.csv"); SP = ["Yeast", "Mouse", "Fly", "Worm", "E. coli"]; BR = ["blend", "global", "MaxSim"]; BN = {"blend": "blend", "global": "global branch", "MaxSim": "MaxSim branch"}
fig, ax = plt.subplots(figsize=(8.6, 6.3)); y = 0; ticks = []; labs = []
for sp in SP:
    for br in BR:
        r = R[(R.set == sp) & (R.branch == br)].iloc[0]; y -= 1
        ax.plot([r.aupr_ctrl, r.aupr_panppi], [y, y], color="#cccccc", lw=1.5, zorder=1)
        ax.hlines(y, r.ctrl_seed_min, r.ctrl_seed_max, color="#8A8F96", lw=2.2, zorder=2); ax.hlines(y, r.panppi_seed_min, r.panppi_seed_max, color="#4A3FA0", lw=2.2, zorder=2)
        ax.plot(r.aupr_panppi, y, "o", color="#4A3FA0", ms=7, zorder=4, label="PanPPI" if (sp == "Yeast" and br == "blend") else None); ax.plot(r.aupr_ctrl, y, "o", color="#8A8F96", ms=6, zorder=3, label="Control without ranking loss" if (sp == "Yeast" and br == "blend") else None)
        ax.text(1.0, y, f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", fontsize=8, va="center", ha="left", transform=ax.get_yaxis_transform())
        ticks.append(y); labs.append(f"{sp} · {BN[br]}" if br == "blend" else f"      {BN[br]}")
    y -= 0.6
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=8.5); ax.set_xlim(max(0.0, 0.43), 0.98); ax.set_xlabel("AUPR (mean of three seeds; whiskers, seed range)"); ax.spines[["top", "right"]].set_visible(False)
ax.text(1.0, 0.985, "PanPPI minus control\n[95% interval]", fontsize=8, fontweight="bold", va="bottom", ha="left", transform=ax.transAxes); ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.62))
plt.tight_layout(); plt.savefig("figS_ablation.png", dpi=190, bbox_inches="tight"); plt.savefig("figS_ablation.svg", bbox_inches="tight"); print("ok")
