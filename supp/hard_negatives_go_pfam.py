"""SUPPLEMENT: annotation-matched hard negatives on the Bernett test set, following LoGoPPI's Supplementary Fig. 4 as closely as the public data allow.
CC-shared = both proteins share at least one DIRECT, positively asserted GO cellular-component annotation (GOA human GAF, aspect C, qualifier without NOT), excluding the CC root (GO:0005575), terms in the subset gocheck_do_not_annotate and
protein-containing complex (GO:0032991) with its is_a descendants (go-basic.obo data-version 2026-07-26). Pfam-shared = both proteins carry an identical Pfam accession (UniProt REST, release 2026_04).
Per replicate (2,000): positives sampled WITH replacement from the eligible positives (sharing) to the number of hard negatives; hard negatives resampled with replacement; size-matched random negatives sampled with replacement from all negatives;
the same positive sample is used for hard and random comparisons. AUPR and AUROC per replicate; mean and 2.5-97.5 percentiles. Models: Bernett-trained PanPPI (each of 3 seeds scored on its own, then averaged), LoGoPPI-Bernett, PLM-interact-Bernett, TUnA-Bernett.
Output hard_negatives_go_pfam.csv."""
import os, sys, gzip, collections, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../paper")
from boot_lib import key
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
A = HERE + "/annot_go_pfam/"; BI = HERE + "/../paper/boot_inputs/"; rng = np.random.default_rng(11); NREP = 2000
# ---- GO
terms = {}; cur = None
for l in open(A + "go-basic.obo"):
    l = l.rstrip("\n")
    if l == "[Term]": cur = {"is_a": [], "subset": set(), "obsolete": False}; continue
    if l.startswith("[") and l != "[Term]": cur = None; continue
    if cur is None or not l: continue
    if l.startswith("id: "): cur["id"] = l[4:]; terms[cur["id"]] = cur
    elif l.startswith("is_a: "): cur["is_a"].append(l[6:].split(" ")[0])
    elif l.startswith("subset: "): cur["subset"].add(l[8:])
    elif l.startswith("is_obsolete: true"): cur["obsolete"] = True
kids = collections.defaultdict(set)
for t, v in terms.items():
    for p in v["is_a"]: kids[p].add(t)
excl = {"GO:0005575", "GO:0032991"}; stack = ["GO:0032991"]
while stack:
    x = stack.pop()
    for c in kids.get(x, ()):
        if c not in excl: excl.add(c); stack.append(c)
excl |= {t for t, v in terms.items() if "gocheck_do_not_annotate" in v["subset"]}; print("excluded CC terms:", len(excl))
cc = collections.defaultdict(set)
for l in gzip.open(A + "goa_human.gaf.gz", "rt"):
    if l.startswith("!"): continue
    f = l.rstrip("\n").split("\t")
    if f[8] != "C" or "NOT" in f[3] or f[4] in excl or f[4] not in terms or terms[f[4]]["obsolete"]: continue
    cc[f[1]].add(f[4])
P = pd.read_csv(A + "pfam_bernett.tsv", sep="\t"); pf = {r.Entry: set(x for x in str(r.Pfam).split(";") if x.startswith("PF")) for r in P.itertuples()}
# ---- scores
S = {}
for s in ("42", "7", "141"):
    d = pd.read_csv(BI + f"runs_ft/e3a_bernett_650M_800_s{s}_h1/scores_bernett.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label", "cal_global", "cal_maxsim"]); d["k"] = key(d.a.to_numpy(), d.b.to_numpy()); S[s] = d.drop_duplicates("k").set_index("k")
base = S["42"].index; y = S["42"].label.to_numpy(); V = {f"PanPPI s{s}": 0.5 * (S[s].cal_global.reindex(base) + S[s].cal_maxsim.reindex(base)).to_numpy() for s in S}
def reindex(df, col, base):
    k = key(df["query"].astype(str).to_numpy(), df["text"].astype(str).to_numpy()); z = pd.Series(df[col].to_numpy(), index=k); z = z[~z.index.duplicated()]; return z.reindex(base).to_numpy()
for tool, f in (("LoGoPPI-Bernett", "logoppi"), ("PLM-interact-Bernett", "plminteract"), ("TUnA-Bernett", "tuna")): V[tool] = reindex(pd.read_csv(BI + f"analysis_out/bernett/{f}_bernett_test.csv", dtype={"query": str, "text": str}), "score", base)
ok = np.all([np.isfinite(v) for v in V.values()], axis=0); a = np.array([k.split("|")[0] for k in base]); b = np.array([k.split("|")[1] for k in base])
SEEDT = [f"PanPPI s{s}" for s in S]; TOOLS = ["PanPPI", "LoGoPPI-Bernett", "PLM-interact-Bernett", "TUnA-Bernett"]
pos = np.where(ok & (y == 1))[0]; neg = np.where(ok & (y == 0))[0]
def shared(ann): return np.array([len(ann.get(x, set()) & ann.get(z, set())) > 0 for x, z in zip(a, b)])
rows = []
for name, ann in (("GO cellular component", cc), ("Pfam domain", pf)):
    sh = shared(ann); hp = pos[sh[pos]]; hn = neg[sh[neg]]; n = len(hn); print(name, "shared positives", len(hp), "shared negatives", n)
    res = {k: {t: [] for t in TOOLS} for k in ("hard", "random")}; resa = {k: {t: [] for t in TOOLS} for k in ("hard", "random")}
    for _ in range(NREP):
        pp = rng.choice(hp, n, replace=True); h = rng.choice(hn, n, replace=True); r = rng.choice(neg, n, replace=True)
        for kind, negs in (("hard", h), ("random", r)):
            idx = np.concatenate([pp, negs]); yy = y[idx]; ap = {t: AP(yy, V[t][idx]) for t in SEEDT + TOOLS[1:]}; au = {t: AU(yy, V[t][idx]) for t in SEEDT + TOOLS[1:]}
            for t in TOOLS[1:]: res[kind][t].append(ap[t]); resa[kind][t].append(au[t])
            res[kind]["PanPPI"].append(np.mean([ap[t] for t in SEEDT])); resa[kind]["PanPPI"].append(np.mean([au[t] for t in SEEDT]))
    for t in TOOLS:
        dh = np.array(res["hard"][t]) - np.array(res["hard"]["PanPPI"])
        rows.append(dict(annotation=name, eligible_positives=len(hp), hard_negatives=n, tool=t, aupr_random=np.mean(res["random"][t]), aupr_random_lo=np.percentile(res["random"][t], 2.5), aupr_random_hi=np.percentile(res["random"][t], 97.5),
                         aupr_hard=np.mean(res["hard"][t]), aupr_hard_lo=np.percentile(res["hard"][t], 2.5), aupr_hard_hi=np.percentile(res["hard"][t], 97.5), auroc_random=np.mean(resa["random"][t]), auroc_hard=np.mean(resa["hard"][t]),
                         diff_vs_panppi_hard=dh.mean(), diff_hard_lo=np.percentile(dh, 2.5), diff_hard_hi=np.percentile(dh, 97.5)))
R = pd.DataFrame(rows); R.round(4).to_csv(HERE + "/hard_negatives_go_pfam.csv", index=False)
print(R[["annotation", "eligible_positives", "hard_negatives", "tool", "aupr_random", "aupr_hard", "diff_vs_panppi_hard", "diff_hard_lo", "diff_hard_hi"]].round(3).to_string(index=False))
