"""SUPPLEMENT: matched-control ablation. PanPPI (E3a: rank loss on proj_m, weight 1.0) vs CTRL (LoGoPPI's recipe trained in OUR harness: identical code, data, splits, selection rule, epochs and calibration; rank_loss_weight 0.0).
Standard cross-species files, 3 seeds each (PanPPI 42/7/141, CTRL 42/7/141), AUPR of the blend (0.5 global + 0.5 MaxSim logits) and of each branch alone. Paired protein-clustered bootstrap (1,000 resamples,
all models on the same resample; each model type averaged over its 3 seeds inside the resample): PanPPI minus CTRL. Test-set noise only; seed range shown separately.
Outputs: ablation_ctrl.csv, ablation_ctrl.png/.svg. Inputs: ../paper/boot_inputs (PanPPI), supp/ctrl_inputs (CTRL)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../paper")
from boot_lib import Boot, key
import boot_lib; boot_lib.B = 1000
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = ["yeast", "mouse", "fly", "worm", "ecoli"]; NM = {"yeast": "Yeast", "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "ecoli": "E. coli"}
PAN = {"42": "e3a_nounc_650M_800_s42_r2", "7": "e3a_nounc_650M_800_s7", "141": "e3a_nounc_650M_800_s141_clean2"}; CTL = {"42": "ctrl_lg_650M_800_s42", "7": "ctrl_lg_650M_800_s7", "141": "ctrl_lg_650M_800_s141"}
def rd(path):
    d = pd.read_csv(path, sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); return d.drop_duplicates("k").set_index("k")
rows = []; seeds_aupr = {}
for sp in SP:
    P = {s: rd(HERE + f"/../paper/boot_inputs/runs_ft/{r}/scores_{sp}.tsv") for s, r in PAN.items()}; C = {s: rd(HERE + f"/ctrl_inputs/runs_ft/{r}/scores_{sp}.tsv") for s, r in CTL.items()}
    base = P["42"].index; y = P["42"].label.to_numpy(); V = {}
    for tag, T in (("PanPPI", P), ("CTRL", C)):
        for s, d in T.items():
            g, m = d.cal_global.reindex(base).to_numpy(), d.cal_maxsim.reindex(base).to_numpy(); V[f"{tag} blend s{s}"] = 0.5 * (g + m); V[f"{tag} global s{s}"] = g; V[f"{tag} MaxSim s{s}"] = m
    ok = np.all([np.isfinite(v) for v in V.values()], axis=0); a = np.array([k.split("|")[0] for k in base[ok]]); b = np.array([k.split("|")[1] for k in base[ok]]); bt = Boot(a, b, y[ok])
    for n, v in V.items(): bt.prep(n, v[ok])
    for br in ("blend", "global", "MaxSim"):
        groups = {"PanPPI": [f"PanPPI {br} s{s}" for s in PAN], "CTRL": [f"CTRL {br} s{s}" for s in PAN]}
        full, out = bt.run(groups, "PanPPI"); o = out[0]
        per = {t: [bt.ap(f"{t} {br} s{s}", np.ones(bt.G)) for s in PAN] for t in ("PanPPI", "CTRL")}
        rows.append(dict(set=NM[sp], branch=br, n=int(ok.sum()), pos=int(y[ok].sum()), aupr_panppi=full["PanPPI"], aupr_ctrl=full["CTRL"], diff=o["diff"], lo=o["lo"], hi=o["hi"], frac_positive=o["frac_positive"],
                         panppi_seed_min=min(per["PanPPI"]), panppi_seed_max=max(per["PanPPI"]), ctrl_seed_min=min(per["CTRL"]), ctrl_seed_max=max(per["CTRL"])))
        print(NM[sp], br, "PanPPI %.4f CTRL %.4f diff %+.4f [%+.4f, %+.4f]" % (full["PanPPI"], full["CTRL"], o["diff"], o["lo"], o["hi"]), flush=True)
R = pd.DataFrame(rows); R.round(5).to_csv(HERE + "/ablation_ctrl.csv", index=False)
PAL = {"PanPPI": "#4A3FA0", "CTRL": "#8A8F96", "blend": "#222222", "global": "#1F5A99", "MaxSim": "#0F8B8D"}
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]}); x = np.arange(len(SP)); w = 0.36; Rb = R[R.branch == "blend"].reset_index(drop=True)
for j, (t, c) in enumerate((("CTRL", PAL["CTRL"]), ("PanPPI", PAL["PanPPI"]))):
    v = Rb[f"aupr_{t.lower()}"].to_numpy(); ax[0].bar(x + (j - 0.5) * w, v, w, color=c, label=("LoGoPPI recipe, our harness (control)" if t == "CTRL" else "PanPPI"))
    lo, hi = Rb[f"{t.lower()}_seed_min"].to_numpy(), Rb[f"{t.lower()}_seed_max"].to_numpy(); ax[0].vlines(x + (j - 0.5) * w, lo, hi, color="black", lw=1.2)
    for xi, vi in zip(x + (j - 0.5) * w, v): ax[0].text(xi, vi + 0.012, f"{vi:.3f}".lstrip("0"), ha="center", fontsize=8.5, fontweight="bold" if t == "PanPPI" else "normal")
ax[0].set_xticks(x); ax[0].set_xticklabels([NM[s] for s in SP]); ax[0].set_ylim(0, 1.2); ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax[0].set_ylabel("AUPR (blend)"); ax[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=1); ax[0].set_title("a  Blend AUPR, mean of three seeds (bars: seed range)", loc="left", fontsize=10.5)
for j, br in enumerate(("blend", "global", "MaxSim")):
    d = R[R.branch == br].reset_index(drop=True); xx = x + (j - 1) * 0.24
    ax[1].errorbar(d["diff"], xx, xerr=[d["diff"] - d["lo"], d["hi"] - d["diff"]], fmt="o", color=PAL[br], capsize=3, label=br, ms=5.5)
ax[1].axvline(0, color="#999999", lw=1); ax[1].set_yticks(x); ax[1].set_yticklabels([NM[s] for s in SP]); ax[1].invert_yaxis(); ax[1].set_xlabel("PanPPI minus control, AUPR (95% paired bootstrap interval)"); ax[1].legend(frameon=False, fontsize=9, title="score", title_fontsize=9, loc="upper right")
ax[1].set_title("b  Effect of the ranking loss, by branch", loc="left", fontsize=10.5)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(HERE + "/ablation_ctrl.png", dpi=170); plt.savefig(HERE + "/ablation_ctrl.svg")
