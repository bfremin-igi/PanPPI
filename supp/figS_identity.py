"""Supplementary Fig. 3: AUPR on the five standard cross-species sets binned by the highest identity of a protein to any human TRAINING protein (pair = the higher of its two proteins). Reads homology_bins.csv (from homology_bins.py).
a: pooled AUPR per bin and tool; b: PanPPI minus LoGoPPI and minus PPLM-PPI with 95% paired protein-clustered bootstrap intervals (pooled and per species); writes figS_identity.png/.svg."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = pd.read_csv("homology_bins.csv"); bins = ["0–20", "20–40", "40–60", "60–80", "80–100"]
COL = {"PanPPI": "#4A3FA0", "LoGoPPI": "#E0566A", "PPLM-PPI": "#E39B2D", "PLM-interact": "#2A9DB5", "ProteomeLM-PPI": "#2E9E5B", "TUnA": "#8B6B4A"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]}); P = D[D.set == "All five"]
for t, c in COL.items():
    y = [P[(P.bin == b) & (P.tool == t)].aupr.iloc[0] for b in bins]; ax[0].plot(range(5), y, "-o", color=c, lw=2.4 if t == "PanPPI" else 1.5, ms=6 if t == "PanPPI" else 4.5, label=t, zorder=5 if t == "PanPPI" else 3)
lab = [f"{b}%\nn+={int(P[(P.bin == b) & (P.tool == 'PanPPI')].n_pos.iloc[0]):,}\nprev. {P[(P.bin == b) & (P.tool == 'PanPPI')].prevalence.iloc[0]:.2f}" for b in bins]
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel("AUPR"); ax[0].set_xlabel("Highest identity to a human training protein (pair: higher of its two proteins)", fontsize=9); ax[0].set_ylim(0.3, 1.0); ax[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right"); ax[0].set_title("a  All five species pooled", loc="left", fontsize=10)
off = {"LoGoPPI": -0.12, "PPLM-PPI": 0.12}
for t, c in (("LoGoPPI", COL["LoGoPPI"]), ("PPLM-PPI", COL["PPLM-PPI"])):
    for k, b in enumerate(bins):
        r = P[(P.bin == b) & (P.tool == t)].iloc[0]; ax[1].errorbar(k + off[t], r.panppi_minus_tool, yerr=[[r.panppi_minus_tool - r.lo], [r.hi - r.panppi_minus_tool]], fmt="o", color=c, capsize=3, ms=6, label=f"PanPPI minus {t}" if k == 0 else None)
ax[1].axhline(0, color="#888", lw=1); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([b + "%" for b in bins], fontsize=9); ax[1].set_ylabel("PanPPI minus comparator, AUPR"); ax[1].set_xlabel("Highest identity to a human training protein", fontsize=9); ax[1].legend(frameon=False, fontsize=8, loc="upper left"); ax[1].set_title("b  Paired difference, 95% bootstrap interval", loc="left", fontsize=10)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figS_identity.png", dpi=200); plt.savefig("figS_identity.svg"); print("ok")
