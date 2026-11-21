"""SUPPLEMENT S14: Bernett test set with the pretrained-human comparators (asterisk). PPLM-PPI* (released weights; analysis_out/bernett_pplm shards) and ProteomeLM-PPI* (lookup in its human all-vs-all file; pairs with a protein
outside the 19,408-protein universe have no score) next to PanPPI (3 seeds) and the Bernett-trained LoGoPPI, PLM-interact and TUnA. AUPR/AUROC on the pairs scored by ALL tools; paired protein-clustered bootstrap (PanPPI minus tool, 1,000 resamples)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../paper")
from boot_lib import Boot, key
import boot_lib; boot_lib.B = 1000
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
BI = HERE + "/../paper/boot_inputs/"; BC = HERE + "/bern_cmp/"
def reindex(df, col, base):
    k = key(df["query"].astype(str).to_numpy(), df["text"].astype(str).to_numpy()); s = pd.Series(df[col].to_numpy(), index=k); s = s[~s.index.duplicated()]; return s.reindex(base).to_numpy()
P = {}
for s in ("42", "7", "141"):
    d = pd.read_csv(BI + f"runs_ft/e3a_bernett_650M_800_s{s}_h1/scores_bernett.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); P[s] = d.drop_duplicates("k").set_index("k")
base = P["42"].index; y = P["42"].label.to_numpy(); V = {f"PanPPI s{s}": 0.5 * (P[s].cal_global.reindex(base) + P[s].cal_maxsim.reindex(base)).to_numpy() for s in P}
for tool, f in (("LoGoPPI-Bernett", "logoppi"), ("PLM-interact-Bernett", "plminteract"), ("TUnA-Bernett", "tuna")):
    V[tool] = reindex(pd.read_csv(BI + f"analysis_out/bernett/{f}_bernett_test.csv", dtype={"query": str, "text": str}), "score", base)
V["PPLM-PPI*"] = reindex(pd.concat([pd.read_csv(BC + f"analysis_out/bernett_pplm/s{i}.csv", dtype={"query": str, "text": str}) for i in range(4)]), "score", base)
V["ProteomeLM-PPI*"] = reindex(pd.read_csv(BC + "analysis_out/bernett/proteomelm_bernett_test.csv", dtype={"query": str, "text": str}), "score", base)
print("coverage (finite scores of %d pairs):" % len(base), {n: int(np.isfinite(v).sum()) for n, v in V.items()})
ok = np.all([np.isfinite(v) for v in V.values()], axis=0); a = np.array([k.split("|")[0] for k in base[ok]]); b = np.array([k.split("|")[1] for k in base[ok]]); bt = Boot(a, b, y[ok])
for n, v in V.items(): bt.prep(n, v[ok])
groups = {"PanPPI": ["PanPPI s42", "PanPPI s7", "PanPPI s141"], **{n: [n] for n in V if not n.startswith("PanPPI")}}
full, out = bt.run(groups, "PanPPI"); yy = y[ok]
rows = [dict(tool="PanPPI (3-seed mean)", aupr=full["PanPPI"], auroc=np.mean([AU(yy, V[f"PanPPI s{s}"][ok]) for s in P]), diff=np.nan, lo=np.nan, hi=np.nan)]
for o in out: rows.append(dict(tool=o["tool"], aupr=o["aupr_tool"], auroc=AU(yy, V[o["tool"]][ok]), diff=o["diff"], lo=o["lo"], hi=o["hi"]))
R = pd.DataFrame(rows); R["n_pairs"] = int(ok.sum()); R["n_pos"] = int(yy.sum()); R.round(4).to_csv(HERE + "/bernett_comparators.csv", index=False); print(R.round(4).to_string(index=False))
