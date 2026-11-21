"""Fig 2d, v6 construct: AUROC (and AUPR at 1:1) on verified direct contacts, human, split by PanPPI training overlap. Reproduces the n's of contact_by_both_homology.csv (1,340 / 1,597) first.
Positives: human PDB-tier L0 variant-A pairs with a direct interface (>=3 residues of each protein within 4.5 A, entries <= 8 chains; fig2/pdb_contact_full.tsv). Negatives: the degree-matched candidate negatives of the
same set (pooled over 3 replicates, unique pairs), UNtrimmed to the subset, so degree is reported as its own baseline (degree-only AUROC). Strata from bern_univ/candidates_with_strata.tsv.gz (1-2 seen, 3-5 new).
Metrics per tool: AUROC; recall at 1/5/20% FPR; paired protein-clustered bootstrap (groups = higher-degree endpoint, 500 draws, all tools on the identical resample). Output fig2/contact_direct_vs_indirect.csv."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
H = "<ROOT>/human_af_screen/"; S = ("s42", "s7", "s141")
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.tier == "pdb") & (B.level == "L0")].copy(); B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in S], axis=0)
PM = pd.read_csv(H + "bern_univ/out3/human/pm_scores.tsv.gz", sep="\t"); B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left"); print("pm missing:", int(B.pm.isna().sum()), "of", len(B)); assert B.pm.notna().all()
C = pd.read_csv(H + "fig2/pdb_contact_full.tsv", sep="\t"); C = C[C.na >= 0]; C["direct"] = (C.na >= 3) & (C.nb >= 3); Dall = C.groupby(["i", "j"]).direct.any().reset_index(); D = Dall[Dall.direct][["i", "j"]]; ND = Dall[~Dall.direct][["i", "j"]]
P0 = B[B.label == 1].drop_duplicates(["i", "j"]); P = pd.concat([P0.merge(D, on=["i", "j"]).assign(kind="direct contact"), P0.merge(ND, on=["i", "j"]).assign(kind="same entry, no direct contact")]); N = B[B.label == 0].drop_duplicates(["i", "j"])
T = pd.read_csv(H + "bern_univ/candidates_with_strata.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "st"]); T = pd.concat([T, T.rename(columns={"uniprot_ac_1": "uniprot_ac_2", "uniprot_ac_2": "uniprot_ac_1"})]).drop_duplicates(["uniprot_ac_1", "uniprot_ac_2"])
P = P.assign(a=P.i.map(lambda x: ids[x]), b=P.j.map(lambda x: ids[x])).merge(T.rename(columns={"uniprot_ac_1": "a", "uniprot_ac_2": "b"}), on=["a", "b"], how="left")
TOOLS = [("PanPPI", "pan"), ("LoGoPPI", "lg"), ("ProteomeLM", "pm"), ("AFM ipSAE", "ipsae"), ("AFM ipTM", "iptm")]
print("direct positives:", len(P), "| st 1-2:", int((P.st <= 2).sum()), "| st 3-5:", int((P.st >= 3).sum()), "| unique negatives:", len(N))
rows = []; rng = np.random.default_rng(7)
for name, m in [(f"{k}, {sn}", (P.kind == k) & sel) for k in ("direct contact", "same entry, no direct contact") for sn, sel in (("seen (st 1-2)", P.st <= 2), ("new (st 3-5)", P.st >= 3))]:
    Pp = P[m]; X = pd.concat([Pp.assign(label=1), N.assign(label=0)]).reset_index(drop=True); y = X.label.to_numpy(); c = pd.concat([X.i, X.j]).value_counts()
    deg = np.log1p(X.i.map(c).to_numpy()) + np.log1p(X.j.map(c).to_numpy()); grp = np.where(X.i.map(c) >= X.j.map(c), X.i, X.j); ug, gi = np.unique(grp, return_inverse=True); idx = [np.where(gi == k)[0] for k in range(len(ug))]
    qs = np.unique(np.quantile(deg[y == 1], np.linspace(0, 1, 21))); qs[0] -= 1e-9; qs[-1] += 1e-9; bn = np.clip(np.digitize(deg, qs) - 1, 0, len(qs) - 2)
    def weights(ix):   # negatives reweighted so their degree distribution matches the positives' (20 degree bins of the positives); positives weight 1
        yy = y[ix]; b = bn[ix]; w = np.ones(len(ix)); pp = np.bincount(b[yy == 1], minlength=len(qs) - 1) / max((yy == 1).sum(), 1); nn = np.bincount(b[yy == 0], minlength=len(qs) - 1) / max((yy == 0).sum(), 1)
        r = np.divide(pp, nn, out=np.zeros_like(pp), where=nn > 0); w[yy == 0] = r[b[yy == 0]]; w[yy == 0] *= (yy == 1).sum() / w[yy == 0].sum(); return w   # negatives total weight = number of positives (1:1, as in Fig 2b,c,e)
    def met(ix):
        yy = y[ix]; out = {}; w = weights(ix)
        for t, col in TOOLS:
            v = X[col].to_numpy()[ix]; out[t] = AU(yy, v); out[t + " dmatch"] = AU(yy, v, sample_weight=w); out[t + " aupr"] = AP(yy, v, sample_weight=w)
            for f in (.01, .05, .2): out[f"{t}@{f}"] = float((v[yy == 1] > np.quantile(v[yy == 0], 1 - f)).mean())
        out["degree"] = AU(yy, deg[ix]); out["degree dmatch"] = AU(yy, deg[ix], sample_weight=w); out["degree aupr"] = AP(yy, deg[ix], sample_weight=w); return out
    base = met(np.arange(len(X))); boots = []
    for _ in range(300):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(ug), len(ug))])
        if y[s].min() == y[s].max(): continue
        boots.append(met(s))
    bd = pd.DataFrame(boots); r = {"segment": name, "n_pos": len(Pp), "n_neg": len(N)}
    for k, v in base.items(): r[k] = v; r[k + "_lo"], r[k + "_hi"] = np.percentile(bd[k], [2.5, 97.5])
    for t in ("LoGoPPI", "AFM ipSAE", "AFM ipTM"):
        for sfx in ("", " dmatch", " aupr"): dd = bd["PanPPI" + sfx] - bd[t + sfx]; r[f"PanPPI-{t}{sfx}"] = base["PanPPI" + sfx] - base[t + sfx]; r[f"PanPPI-{t}{sfx}_lo"], r[f"PanPPI-{t}{sfx}_hi"] = np.percentile(dd, [2.5, 97.5])
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(H + "fig2/contact_direct_vs_indirect.csv", index=False)
print(R[["segment", "n_pos"] + [t for t, _ in TOOLS] + ["degree"] + [t + " dmatch" for t, _ in TOOLS] + ["degree dmatch"] + [f"{t}@0.05" for t, _ in TOOLS] + [f"PanPPI-{t}{x}" for x in (" dmatch",) for t in ("LoGoPPI", "AFM ipSAE", "AFM ipTM")]].round(3).T.to_string())
