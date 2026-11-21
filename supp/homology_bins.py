"""SUPPLEMENT S6: accuracy stratified by sequence identity to the human TRAINING set, standard cross-species sets (follows LoGoPPI Supplementary Fig. 2).
Per test protein: highest identity to any human training protein (mmseqs, analysis_out/stdhom/hits.m8; no hit = 0). Pair level = higher of the two proteins. Bins [0-20], (20-40], (40-60], (60-80], (80-100] %.
AUPR per bin on pairs scored by ALL tools, per species and pooled over the five species. PanPPI = mean AUPR of its 3 seeds, TUnA (3 released seeds) likewise, others single models.
Paired protein-clustered bootstrap (1,000 resamples, groups = higher-degree protein inside the bin) for PanPPI minus LoGoPPI and PanPPI minus PPLM-PPI. Prevalence per bin is reported because AUPR depends on it.
Run: python3 -I homology_bins.py   (inputs: ../paper/boot_inputs, ../human_af_screen/pm/results, supp/hits.m8)"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../paper")
from boot_lib import Boot, key
import boot_lib; boot_lib.B = 1000
BI = HERE + "/../paper/boot_inputs/"; HS = HERE + "/../human_af_screen/"
SP = ["yeast", "mouse", "fly", "worm", "ecoli"]; NM = {"yeast": "Yeast", "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "ecoli": "E. coli", "pooled": "All five"}
BINS = [(0.0, 0.2, "0–20"), (0.2, 0.4, "20–40"), (0.4, 0.6, "40–60"), (0.6, 0.8, "60–80"), (0.8, 1.0001, "80–100")]
RUNS = {"s42": "e3a_nounc_650M_800_s42_r2", "s7": "e3a_nounc_650M_800_s7", "s141": "e3a_nounc_650M_800_s141_clean2"}
hits = pd.read_csv(HERE + "/hits.m8", sep="\t", header=None, names=["q", "t", "pid", "qcov", "tcov", "ev", "bits"], dtype={"q": str, "t": str}); maxid = (hits.groupby("q").pid.max() / 100.0).to_dict()
def reindex(df, ka, kb, col, base):
    k = key(df[ka].astype(str).to_numpy(), df[kb].astype(str).to_numpy()); s = pd.Series(df[col].to_numpy(), index=k); s = s[~s.index.duplicated()]; return s.reindex(base).to_numpy()
def load(sp):
    P = {}
    for s, r in RUNS.items():
        d = pd.read_csv(BI + f"runs_ft/{r}/scores_{sp}.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); P[s] = d.drop_duplicates("k").set_index("k")
    base = P["s42"].index; y = P["s42"].label.to_numpy(); V = {f"PanPPI {s}": 0.5 * (P[s].cal_global.reindex(base) + P[s].cal_maxsim.reindex(base)).to_numpy() for s in P}
    for tool, f in (("LoGoPPI", "logoppi"), ("PPLM-PPI", "pplm"), ("PLM-interact", "plminteract"), ("TUnA s47", "tuna_s47"), ("TUnA s94", "tuna_s94"), ("TUnA s141", "tuna_s141")):
        V[tool] = reindex(pd.read_csv(BI + f"analysis_out/xsp_bench/{f}/{sp}.csv", dtype={"query": str, "text": str}), "query", "text", "score", base)
    V["ProteomeLM-PPI"] = reindex(pd.read_csv(HS + f"pm/results/{sp}.tsv", sep="\t", dtype={"query": str, "text": str}), "query", "text", "supervised_score", base)
    ok = np.all([np.isfinite(v) for v in V.values()], axis=0); a = np.array([k.split("|")[0] for k in base]); b = np.array([k.split("|")[1] for k in base])
    lvl = np.maximum(pd.Series(a).map(maxid).fillna(0.0).to_numpy(), pd.Series(b).map(maxid).fillna(0.0).to_numpy())
    return dict(a=a[ok], b=b[ok], y=y[ok], lvl=lvl[ok], V={n: v[ok] for n, v in V.items()}, sp=np.full(ok.sum(), sp))
GROUPS = {"PanPPI": ["PanPPI s42", "PanPPI s7", "PanPPI s141"], "LoGoPPI": ["LoGoPPI"], "PPLM-PPI": ["PPLM-PPI"], "PLM-interact": ["PLM-interact"], "ProteomeLM-PPI": ["ProteomeLM-PPI"], "TUnA": ["TUnA s47", "TUnA s94", "TUnA s141"]}
D = {sp: load(sp) for sp in SP}
D["pooled"] = {k: (np.concatenate([D[s][k] for s in SP]) if k != "V" else {n: np.concatenate([D[s]["V"][n] for s in SP]) for n in D["yeast"]["V"]}) for k in ("a", "b", "y", "lvl", "V", "sp")}
rows = []
for sp, d in D.items():
    for lo, hi, lab in BINS:
        m = (d["lvl"] >= lo) & (d["lvl"] < hi) if lo > 0 else (d["lvl"] < hi); y = d["y"][m]
        r0 = dict(set=NM[sp], bin=lab, n_pairs=int(m.sum()), n_pos=int(y.sum()), prevalence=float(y.mean()) if m.sum() else np.nan)
        if m.sum() < 50 or y.sum() < 10 or y.sum() == len(y): rows.append(dict(r0, tool="(too few)", aupr=np.nan)); continue
        a, b = d["a"][m], d["b"][m]
        # pooled: species-prefixed ids are already distinct (taxid prefixes), so grouping stays within species
        bt = Boot(a, b, y)
        for n, v in d["V"].items(): bt.prep(n, v[m])
        full, out = bt.run(GROUPS, "PanPPI"); d_ = {o["tool"]: o for o in out}
        for t, v in full.items():
            r = dict(r0, tool=t, aupr=v)
            if t in ("LoGoPPI", "PPLM-PPI"): r.update(panppi_minus_tool=d_[t]["diff"], lo=d_[t]["lo"], hi=d_[t]["hi"])
            rows.append(r)
        print(NM[sp], lab, "n=%d pos=%d prev=%.3f" % (m.sum(), y.sum(), y.mean()), " ".join("%s %.3f" % (t, v) for t, v in full.items()), flush=True)
R = pd.DataFrame(rows); R.round(5).to_csv(HERE + "/homology_bins.csv", index=False)
# bin occupancy of proteins for the legend
prot = pd.Series({p: maxid.get(p, 0.0) for s in SP for p in np.r_[D[s]["a"], D[s]["b"]]}); print("\nprotein-level best identity, share of proteins per bin:", {lab: round(float(((prot >= lo) & (prot < hi)).mean()), 3) for lo, hi, lab in BINS})
