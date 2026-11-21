"""Supplementary Fig. 5: selection-split AUPR of the calibrated blend after each epoch for the nine training runs (PanPPI and control, human training; PanPPI, Bernett training), best epoch marked. Reads training_logs_summary.txt
(copied from runs_ft/<run>/output/results.txt). Early stopping: patience 3 epochs without improvement of this metric; ceiling 14 epochs; no run reached the ceiling."""
import re, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, pandas as pd
T = open("training_logs_summary.txt").read().split("### ")[1:]; R = {}
for blk in T:
    name, body = blk.split("\n", 1); ep = [l for l in body.split("\n") if l.startswith("epoch ")]
    sel = [float(re.search(r"SEL\[[^\]]*\] ([0-9.]+)", l).group(1)) for l in ep]; best = int(re.search(r"best ep (\d+)", ep[-1]).group(1)); R[name] = (sel, best)
G = {"human": [("e3a_nounc_650M_800_s42_r2", "PanPPI", "#4A3FA0", "42"), ("e3a_nounc_650M_800_s7", "PanPPI", "#4A3FA0", "7"), ("e3a_nounc_650M_800_s141_clean2", "PanPPI", "#4A3FA0", "141"), ("ctrl_lg_650M_800_s42", "Control", "#8A8F96", "42"), ("ctrl_lg_650M_800_s7", "Control", "#8A8F96", "7"), ("ctrl_lg_650M_800_s141", "Control", "#8A8F96", "141")],
     "bernett": [("e3a_bernett_650M_800_s42_h1", "PanPPI", "#4A3FA0", "42"), ("e3a_bernett_650M_800_s7_h1", "PanPPI", "#4A3FA0", "7"), ("e3a_bernett_650M_800_s141_h1", "PanPPI", "#4A3FA0", "141")]}
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1)); LS = {"42": "-", "7": "--", "141": ":"}
for a, (k, title) in zip(ax, (("human", "a  Human training set (selection split)"), ("bernett", "b  Bernett training set (selection split)"))):
    for run, lab, c, sd in G[k]:
        sel, best = R[run]; x = list(range(1, len(sel) + 1)); a.plot(x, sel, LS[sd], color=c, lw=1.6, marker="o", ms=3.5, label=f"{lab}, seed {sd}")
        a.plot(best, sel[best - 1], marker="*", color=c, ms=13, mec="black", mew=0.6, zorder=5)
    a.set_xlabel("Epoch"); a.set_ylabel("AUPR of the calibrated blend"); a.set_title(title, loc="left", fontsize=10); a.set_xticks(range(1, 12)); a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
ax[0].set_ylim(0.80, 0.965); ax[1].set_ylim(0.60, 0.70)
plt.tight_layout(); plt.savefig("figS_training.png", dpi=190); plt.savefig("figS_training.svg"); print("ok")
