"""SUPPLEMENT: annotation-matched hard negatives on the Bernett test set (counterpart of LoGoPPI Supplementary Fig. 4). Models trained on the Bernett split: PanPPI (mean of 3 seeds), LoGoPPI-Bernett, PLM-interact-Bernett, TUnA-Bernett.
DESIGN 1 (as in LoGoPPI Supplementary Fig. 4): positives are drawn from the positives that share the annotation, so positives and negatives are alike in sharing; DESIGN 2: positives drawn from all positives (more adversarial). Hard negatives = negative pairs whose two proteins share at least one UniProt cellular-component keyword (Nucleus, Cytoplasm, Mitochondrion, Cell membrane, Endoplasmic reticulum, Golgi apparatus, Secreted, Cytoskeleton, Cell projection,
Chromosome, Cytoplasmic vesicle, Endosome, Lysosome, Peroxisome, Cell junction, Extracellular matrix, Nucleolus); strict variant: at least two shared keywords. For each variant the hard set (size n) is compared with a random size-matched set of negatives;
positives are subsampled to n (1:1 prevalence), 200 repeats, so AUPR is at 50% prevalence and comparable between sets. Pairs scored by all tools. Output hard_negatives.csv."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../paper")
from boot_lib import key
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
BI = HERE + "/../paper/boot_inputs/"; rng = np.random.default_rng(5)
def reindex(df, col, base):
    k = key(df["query"].astype(str).to_numpy(), df["text"].astype(str).to_numpy()); s = pd.Series(df[col].to_numpy(), index=k); s = s[~s.index.duplicated()]; return s.reindex(base).to_numpy()
P = {}
for s in ("42", "7", "141"):
    d = pd.read_csv(BI + f"runs_ft/e3a_bernett_650M_800_s{s}_h1/scores_bernett.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); P[s] = d.drop_duplicates("k").set_index("k")
base = P["42"].index; y = P["42"].label.to_numpy(); V = {f"PanPPI s{s}": 0.5 * (P[s].cal_global.reindex(base) + P[s].cal_maxsim.reindex(base)).to_numpy() for s in P}
for tool, f in (("LoGoPPI-Bernett", "logoppi"), ("PLM-interact-Bernett", "plminteract"), ("TUnA-Bernett", "tuna")): V[tool] = reindex(pd.read_csv(BI + f"analysis_out/bernett/{f}_bernett_test.csv", dtype={"query": str, "text": str}), "score", base)
ok = np.all([np.isfinite(v) for v in V.values()], axis=0); a = np.array([k.split("|")[0] for k in base]); b = np.array([k.split("|")[1] for k in base]); print("pairs scored by all:", int(ok.sum()), "of", len(base))
CC = {"Nucleus", "Cytoplasm", "Mitochondrion", "Cell membrane", "Endoplasmic reticulum", "Golgi apparatus", "Secreted", "Cytoskeleton", "Cell projection", "Chromosome", "Cytoplasmic vesicle", "Endosome", "Lysosome", "Peroxisome", "Cell junction", "Extracellular matrix", "Nucleolus"}
K = pd.read_csv(HERE + "/../human_af_screen/fig2/annot/uniprot_human_keywords.tsv", sep="\t"); cc = {r.Entry: set(str(r.Keywords).split(";")) & CC for r in K.itertuples()}
share = np.array([len(cc.get(x, set()) & cc.get(z, set())) for x, z in zip(a, b)])
SEEDT = [f"PanPPI s{s}" for s in P]; TOOLS = SEEDT + ["LoGoPPI-Bernett", "PLM-interact-Bernett", "TUnA-Bernett"]
pos = np.where(ok & (y == 1))[0]; neg = np.where(ok & (y == 0))[0]; rows = []
print("share>=1: positives %.3f negatives %.3f | share>=2: positives %.3f negatives %.3f" % ((share[pos] >= 1).mean(), (share[neg] >= 1).mean(), (share[pos] >= 2).mean(), (share[neg] >= 2).mean()))
def evaluate(label, hard_idx, pool):
    n = len(hard_idx); res = {t: {"aupr": [], "auroc": []} for t in TOOLS}; res_r = {t: {"aupr": [], "auroc": []} for t in TOOLS}
    for _ in range(200):
        pp = rng.choice(pool, n, replace=False); rn = rng.choice(neg, n, replace=False)
        for nm, negs, store in (("hard", hard_idx, res), ("random", rn, res_r)):
            idx = np.concatenate([pp, negs]); yy = y[idx]
            for t in TOOLS: store[t]["aupr"].append(AP(yy, V[t][idx])); store[t]["auroc"].append(AU(yy, V[t][idx]))
    for st in (res, res_r): st["PanPPI"] = {m: np.mean([st[t][m] for t in SEEDT], axis=0) for m in ("aupr", "auroc")}
    for t in ["PanPPI"] + TOOLS:
        dh = np.array(res[t]["aupr"]) - np.array(res["PanPPI"]["aupr"]); dr = np.array(res_r[t]["aupr"]) - np.array(res_r["PanPPI"]["aupr"])
        rows.append(dict(variant=label, n_per_class=n, tool=t, aupr_random=np.mean(res_r[t]["aupr"]), aupr_hard=np.mean(res[t]["aupr"]), auroc_random=np.mean(res_r[t]["auroc"]), auroc_hard=np.mean(res[t]["auroc"]), aupr_hard_lo=np.percentile(res[t]["aupr"], 2.5), aupr_hard_hi=np.percentile(res[t]["aupr"], 97.5), aupr_random_lo=np.percentile(res_r[t]["aupr"], 2.5), aupr_random_hi=np.percentile(res_r[t]["aupr"], 97.5), diff_vs_panppi_hard=dh.mean(), diff_hard_lo=np.percentile(dh, 2.5), diff_hard_hi=np.percentile(dh, 97.5), diff_vs_panppi_random=dr.mean(), diff_random_lo=np.percentile(dr, 2.5), diff_random_hi=np.percentile(dr, 97.5)))
for design, poolf in (("matched positives (as LoGoPPI)", lambda k: pos[share[pos] >= k]), ("all positives", lambda k: pos)):
    for k, nm in ((1, "share >= 1 compartment keyword"), (2, "share >= 2 compartment keywords")):
        evaluate(design + " | " + nm, neg[share[neg] >= k], poolf(k)); print(design, nm, "eligible positives", len(poolf(k)), "hard negatives", int((share[neg] >= k).sum()))
R = pd.DataFrame(rows); R.round(4).to_csv(HERE + "/hard_negatives.csv", index=False); print(R[["variant", "n_per_class", "tool", "aupr_random", "aupr_hard", "auroc_random", "auroc_hard"]].round(3).to_string(index=False))
