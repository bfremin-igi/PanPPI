"""Supplementary figure on Figure 2 (retention of AFM-confident pairs): a other AFM-confidence definitions; b contribution of each part of PanPPI and simple baselines; c hub removal; d the three seeds; e pairs the model has not seen in training.
Reads supp2_threshold.csv, supp2_seeds.csv, supp2_novelty.csv (supp2_compute.py), ../retention_extended.csv, ../retention_hub_trim.csv. Writes supp2_figure.png/.svg."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
PAN, LG, GL, MS, LEN = "#4A3FA0", "#E0566A", "#2A9DB5", "#E39B2D", "#9AA0A6"
SP = ["Human (new pairs)", "Yeast", "Mouse", "Fly", "Worm", "E. coli"]; SHORT = ["Human\n(new pairs)", "Yeast", "Mouse", "Fly", "Worm", "E. coli"]
T = pd.read_csv("supp2_threshold.csv"); Sd = pd.read_csv("supp2_seeds.csv"); N = pd.read_csv("supp2_novelty.csv"); X = pd.read_csv("../retention_extended.csv"); Hb = pd.read_csv("../retention_hub_trim.csv")
fig = plt.figure(figsize=(13.2, 9.2)); gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=1.4)
ax_a = fig.add_subplot(gs[0, 0:3]); ax_b = fig.add_subplot(gs[0, 3:6]); ax_c = fig.add_subplot(gs[1, 0:2]); ax_d = fig.add_subplot(gs[1, 2:4]); ax_e = fig.add_subplot(gs[1, 4:6])
# ---- a
defs = ["paper label (ipSAE >= 0.6 and pDockQ2 >= 0.23)", "ipSAE >= 0.5", "ipSAE >= 0.7", "ipSAE >= 0.8", "ipTM >= 0.7"]; dl = ["paper label", "ipSAE ≥ 0.5", "ipSAE ≥ 0.7", "ipSAE ≥ 0.8", "ipTM ≥ 0.7"]
shade = ["#2B2470", PAN, "#7A70C9", "#B3ADE3", "#D5D1F0"]; t1 = T[T.fraction == 0.01]; w = 0.15
for k, d in enumerate(defs):
    for si, sp in enumerate(SP):
        r = t1[(t1.species == sp) & (t1.definition == d)]
        if r.empty: continue
        x = si + (k - 2) * w; ax_a.bar(x, r.PanPPI.iloc[0] * 100, w * 0.92, color=shade[k], lw=0); ax_a.plot(x, r.LoGoPPI.iloc[0] * 100, "o", ms=3.8, mfc="white", mec=LG, mew=1.2)
ax_a.set_xticks(range(6)); ax_a.set_xticklabels(SHORT); ax_a.set_ylim(0, 125); ax_a.set_yticks(range(0, 101, 20)); ax_a.set_ylabel("AFM-confident pairs in top 1% (%)"); ax_a.set_title("a  Other definitions of 'AFM-confident'", loc="left", fontsize=10, fontweight="bold")
ax_a.legend(handles=[Patch(color=shade[k], label=dl[k]) for k in range(5)] + [Line2D([], [], marker="o", ls="", mfc="white", mec=LG, label="LoGoPPI")], fontsize=7.5, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
# ---- b
cols = [("PanPPI blend", PAN, "PanPPI"), ("PanPPI global only", GL, "global only"), ("PanPPI MaxSim only", MS, "MaxSim only"), ("LoGoPPI", LG, "LoGoPPI"), ("Length product", LEN, "length product")]
x1 = X[X.fraction == 0.01].set_index("set"); w = 0.15
for k, (c, col, lab) in enumerate(cols):
    for si, sp in enumerate(SP): ax_b.bar(si + (k - 2) * w, x1.loc[sp, c] * 100, w * 0.92, color=col, lw=0)
for si in range(6): ax_b.plot([si - 0.42, si + 0.42], [1, 1], color="black", lw=0.8, ls=":")
ax_b.set_xticks(range(6)); ax_b.set_xticklabels(SHORT); ax_b.set_ylim(0, 125); ax_b.set_yticks(range(0, 101, 20)); ax_b.set_ylabel("AFM-confident pairs in top 1% (%)"); ax_b.set_title("b  What each part contributes", loc="left", fontsize=10, fontweight="bold")
ax_b.legend(handles=[Patch(color=c, label=l) for _, c, l in cols] + [Line2D([], [], color="black", ls=":", label="random (1%)")], fontsize=7.5, ncol=3, loc="upper center", frameon=False)
# ---- c
for sp, col in zip(SP[:5], ["#2B2470", "#4A3FA0", "#7A70C9", "#2A9DB5", "#E39B2D"]):
    h = Hb[(Hb.set == sp) & (Hb.fraction == 0.01)].sort_values("hub_fraction_removed"); ax_c.plot(h.hub_fraction_removed * 100, h.PanPPI * 100, "-o", ms=4, color=col, label=sp.replace(" (new pairs)", ""))
    ax_c.plot(h.hub_fraction_removed * 100, h.LoGoPPI * 100, "--", color=col, lw=1)
ax_c.set_xlabel("proteins removed as hubs (top %)"); ax_c.set_ylabel("AFM-confident pairs in top 1% (%)"); ax_c.set_xticks([0, 1, 5, 10]); ax_c.set_ylim(15, 95); ax_c.set_title("c  Hub removal", loc="left", fontsize=10, fontweight="bold")
ax_c.legend(fontsize=7.2, frameon=False, ncol=3, loc="lower center", title="solid PanPPI, dashed LoGoPPI", title_fontsize=7, columnspacing=0.8, handlelength=1.4)
# ---- d
s1 = Sd[Sd.fraction == 0.01].set_index("species")
for si, sp in enumerate(SP):
    v = np.array([s1.loc[sp, f"PanPPI_{s}"] for s in ("s42", "s7", "s141")]) * 100; ax_d.bar(si, v.mean(), 0.55, color="#D5D1F0", lw=0); ax_d.plot([si] * 3 + np.array([-0.12, 0, 0.12]), v, "o", ms=5, color=PAN); ax_d.plot(si + 0.34, s1.loc[sp, "LoGoPPI"] * 100, "D", ms=5, color=LG)
ax_d.set_xticks(range(6)); ax_d.set_xticklabels(SHORT, fontsize=8); ax_d.set_ylim(0, 125); ax_d.set_yticks(range(0, 101, 20)); ax_d.set_ylabel("AFM-confident pairs in top 1% (%)"); ax_d.set_title("d  The three PanPPI seeds", loc="left", fontsize=10, fontweight="bold")
ax_d.legend(handles=[Line2D([], [], marker="o", ls="", color=PAN, label="PanPPI seed"), Line2D([], [], marker="D", ls="", color=LG, label="LoGoPPI")], fontsize=7.5, frameon=False, loc="upper center", ncol=2)
# ---- e
n1 = N[N.fraction == 0.01]
G = [("Human (new pairs)", [("pair or homologous pair in training", "#D5D1F0"), ("new pair, both proteins have a training homolog", "#B3ADE3"), ("new pair, one protein without a homolog", "#2B2470")], "Human"),
     ("Yeast", [("at least one protein has a training homolog", "#B3ADE3"), ("neither protein has a training homolog", "#2B2470")], "Yeast"), ("Fly", [("at least one protein has a training homolog", "#B3ADE3"), ("neither protein has a training homolog", "#2B2470")], "Fly"),
     ("Worm", [("at least one protein has a training homolog", "#B3ADE3"), ("neither protein has a training homolog", "#2B2470")], "Worm"), ("Mouse", [("at least one protein has a training homolog", "#B3ADE3"), ("neither protein has a training homolog", "#2B2470")], "Mouse"),
     ("E. coli", [("at least one protein has a training homolog", "#B3ADE3"), ("neither protein has a training homolog", "#2B2470")], "E. coli")]
pos = 0; ticks, tl, centers = [], [], []
for sp, items, lab in G:
    x0 = pos
    for g, col in items:
        r = n1[(n1.species == sp) & (n1.group == g)].iloc[0]
        if pd.notna(r.PanPPI): ax_e.bar(pos, r.PanPPI * 100, 0.8, color=col, lw=0); ax_e.plot([pos, pos], [r.PanPPI_lo * 100, r.PanPPI_hi * 100], color="black", lw=1); ax_e.plot(pos, r.LoGoPPI * 100, "o", ms=4.5, mfc="white", mec=LG, mew=1.3)
        else: ax_e.text(pos, 4, "n too\nsmall", ha="center", fontsize=6.5, rotation=90)
        ticks.append(pos); tl.append(f"{int(r.n):,}"); pos += 1
    centers.append(((x0 + pos - 1) / 2, lab)); pos += 0.8
ax_e.set_xticks(ticks); ax_e.set_xticklabels(tl, rotation=90, fontsize=6.5); ax_e.tick_params(axis="x", length=2, pad=1)
for c, lab in centers: ax_e.text(c, -0.30, lab, transform=ax_e.get_xaxis_transform(), ha="center", va="top", fontsize=8)
ax_e.text(-0.02, -0.12, "n", transform=ax_e.transAxes, ha="right", fontsize=6.5)
ax_e.set_ylim(0, 125); ax_e.set_yticks(range(0, 101, 20)); ax_e.set_ylabel("AFM-confident pairs in top 1% (%)"); ax_e.set_title("e  Pairs not seen in training", loc="left", fontsize=10, fontweight="bold")
ax_e.legend(handles=[Patch(color="#D5D1F0", label="human pair (or homologous pair) in training"), Patch(color="#B3ADE3", label="new pair, proteins have a training homolog"), Patch(color="#2B2470", label="no training homolog for one or both proteins"), Line2D([], [], marker="o", ls="", mfc="white", mec=LG, label="LoGoPPI")], fontsize=6.6, frameon=False, loc="upper center", ncol=1)
fig.savefig("supp2_figure.png", dpi=170, bbox_inches="tight"); fig.savefig("supp2_figure.svg", bbox_inches="tight"); print("ok")
