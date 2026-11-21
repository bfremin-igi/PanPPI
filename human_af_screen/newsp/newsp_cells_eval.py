#!/usr/bin/env python3
"""Evaluation of the multi-species degree-matched cells with the definitions of bern_univ/eval_cand.py, eval_cand_match_afm.py and fig2/precision_ci.py (no ProteomeLM): per cell AUROC/AUPR (PanPPI mean over seeds; LoGoPPI; AFM ipSAE, ipTM, flag) on balanced
replicates, degree-only AUROC, inclusion (negatives >= 80% of retained positives and degree-only AUROC <= 0.60); precision of calls (AFM flag count, matched top-k, 95th percentile of the tool's negatives) pooled over the kept cells and replicates
with a protein-clustered bootstrap (300 draws; thresholds recomputed per resample); and the no-trimming check (all s700 positives kept, negatives degree-reweighted). Outputs analysis_out/newsp/newsp_cells_results.csv, newsp_precision.csv, newsp_noTrim.csv."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
import os
SPS = ["soybean", "arabidopsis", "rat", "rice", "zebrafish"]; SEEDS = tuple(sd for sd in ("s42", "s7", "s141") if all(os.path.exists(f"analysis_out/newsp/{sp}/cells_panppi_{sd}.csv") for sp in SPS)); print("PanPPI seeds used:", SEEDS, flush=True); assert SEEDS, "no complete PanPPI seed"; rng = np.random.default_rng(3); BDRAW = 300; res_rows = []; SETS = []; nt_rows = []
def attach(df, O, with_lg=True):
    for s in SEEDS:
        z = pd.read_csv(f"{O}/cells_panppi_{s}.csv", dtype={"protein_a": str, "protein_b": str}).rename(columns={"protein_a": "a", "protein_b": "b", "g": f"g_{s}", "m": f"m_{s}"}); df = df.merge(z, on=["a", "b"], how="left"); df[f"bl_{s}"] = 0.5 * (df[f"g_{s}"] + df[f"m_{s}"])
    L = pd.read_csv(f"{O}/cells_logoppi.csv", dtype={"query": str, "text": str}); P = pd.read_csv(f"{O}/cell_pairs.csv", dtype=str); assert (L["query"].to_numpy() == P.protein_a.to_numpy()).all() and (L["text"].to_numpy() == P.protein_b.to_numpy()).all()
    df = df.merge(pd.DataFrame({"a": P.protein_a, "b": P.protein_b, "lg": L["score"].to_numpy()}), on=["a", "b"], how="left"); assert df[[f"bl_{s}" for s in SEEDS] + ["lg"]].notna().all().all(), "missing scores"; df["pan"] = df[[f"bl_{s}" for s in SEEDS]].mean(axis=1); return df
for sp in SPS:
    O = f"analysis_out/newsp/{sp}"; meta = json.load(open(f"{O}/cell_meta.json"))
    for m in meta:
        if "note" in m: print(sp, m["level"], "skipped:", m["note"], m["n_pos"]); continue
        G = pd.read_csv(f"{O}/cells/{m['tier']}_{m['level']}_{m['variant']}.tsv.gz", sep="\t", dtype={"a": str, "b": str}); G = attach(G, O); P = G[G.label == 1]; reps = sorted(set(G[G.label == 0].rep)); frac = min(m["neg_per_rep"]) / m["n_pos"]; dg = []; res = {}; cand_sets = []
        for r in reps:
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); D = pd.concat([P.sample(k, random_state=r + 1) if len(P) > k else P, N.sample(k, random_state=r + 7) if len(N) > k else N]).reset_index(drop=True); y = D.label.to_numpy()
            c = pd.concat([D.i, D.j]).value_counts(); dg.append(AU(y, np.log1p(D.i.map(c).to_numpy()) + np.log1p(D.j.map(c).to_numpy())))
            T = {"PanPPI blend": [D[f"bl_{s}"] for s in SEEDS], "PanPPI global": [D[f"g_{s}"] for s in SEEDS], "LoGoPPI": [D.lg], "AFM ipSAE": [D.ipsae], "AFM ipTM": [D.iptm]}
            for nm, vs in T.items(): res.setdefault(nm, []).append((np.mean([AU(y, v) for v in vs]), np.mean([AP(y, v) for v in vs])))
            res.setdefault("AFM confident flag", []).append((AU(y, D.afm_pass), AP(y, D.afm_pass)))
            if True:
                grp = np.where(D.i.map(c) >= D.j.map(c), D.i, D.j); u, inv = np.unique(grp, return_inverse=True); order = np.argsort(inv, kind="stable"); bd = np.r_[0, np.cumsum(np.bincount(inv))]
                cand_sets.append(dict(sp=sp, y=y, pan=D.pan.to_numpy(), lg=D.lg.to_numpy(), afm=D.afm_pass.to_numpy().astype(bool), groups=[order[bd[q]:bd[q + 1]] for q in range(len(u))], nu=len(u)))
        deg = float(np.mean(dg))
        if frac >= 0.8 and deg <= 0.60: SETS.extend(cand_sets)
        out = {"seeds": len(SEEDS), "species": sp, "tier": m["tier"], "level": m["level"], "variant": m["variant"], "n_pos": m["n_pos"], "n_pos_total": m["n_pos_total"], "retained": m["retained"], "neg_frac": round(frac, 2), "deg_only_auroc": round(deg, 3), "kept": bool(frac >= 0.8 and deg <= 0.60)}
        for nm, v in res.items(): out[nm + " auroc"] = np.mean([x[0] for x in v]); out[nm + " aupr"] = np.mean([x[1] for x in v])
        res_rows.append(out); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k in ("species", "level", "n_pos", "retained", "neg_frac", "deg_only_auroc", "kept", "PanPPI blend aupr", "LoGoPPI aupr", "AFM ipSAE aupr")}, flush=True)
    # ---- no-trim reweighted check (L0): all s700 positives, sampled low-confidence candidates reweighted by degree bins
    E = attach(pd.read_csv(f"{O}/reweight_set.tsv.gz", sep="\t", dtype={"a": str, "b": str}), O); k = pd.concat([E.i, E.j]).value_counts(); E["d"] = np.log1p(E.i.map(k).to_numpy()) + np.log1p(E.j.map(k).to_numpy())
    qs = np.quantile(E.d[E.label == 1], np.linspace(0, 1, 21)); qs[0], qs[-1] = -np.inf, np.inf; b = np.clip(np.searchsorted(qs, E.d, side="right") - 1, 0, 19); pp = np.bincount(b[E.label == 1], minlength=20) / (E.label == 1).sum(); pn = np.bincount(b[E.label == 0], minlength=20) / max((E.label == 0).sum(), 1)
    w = np.where(E.label == 1, 1.0, np.where(pn[b] > 0, pp[b] / np.maximum(pn[b], 1e-12), 0.0)); neg = E.label.to_numpy() == 0; w[neg] *= (E.label == 1).sum() / w[neg].sum()
    for nm, col in (("PanPPI", "pan"), ("LoGoPPI", "lg"), ("AFM ipSAE", "ipsae"), ("AFM ipTM", "iptm")): nt_rows.append(dict(species=sp, n_pos=int((E.label == 1).sum()), n_neg=int(neg.sum()), tool=nm, aupr_reweighted=AP(E.label, E[col], sample_weight=w), aupr_unweighted_raw=AP(E.label, E[col]), degree_only_auroc_after_weighting=AU(E.label, E.d, sample_weight=w)))
R = pd.DataFrame(res_rows); R.round(4).to_csv("analysis_out/newsp/newsp_cells_results.csv", index=False); pd.DataFrame(nt_rows).round(4).to_csv("analysis_out/newsp/newsp_noTrim.csv", index=False)
def counts(Dd, ix):
    y, pan, lg, afm = Dd["y"][ix], Dd["pan"][ix], Dd["lg"][ix], Dd["afm"][ix]; kk = int(afm.sum()); out = {"afm": (kk, int(y[afm].sum()))}
    for nm, sc in (("pan", pan), ("lg", lg)):
        top = np.argsort(-sc, kind="stable")[:kk]; out[nm + "_match"] = (kk, int(y[top].sum())); thr = np.quantile(sc[y == 0], 0.95); call = sc > thr; out[nm + "_top5"] = (int(call.sum()), int(y[call].sum()))
    return out
def pooled(draw, sel=None):
    tot = {}
    for Dd in SETS:
        if sel is not None and Dd["sp"] not in sel: continue
        ix = np.arange(len(Dd["y"])) if not draw else np.concatenate([Dd["groups"][q] for q in rng.integers(0, Dd["nu"], Dd["nu"])])
        for k_, (c_, t_) in counts(Dd, ix).items(): a = tot.setdefault(k_, [0, 0]); a[0] += c_; a[1] += t_
    return {k_: (v[1] / v[0] if v[0] else np.nan, v[0], v[1]) for k_, v in tot.items()}
rows = []
for label, sel in [("all five species", None)] + [(s, {s}) for s in SPS]:
    if not any((Dd["sp"] in (sel or SPS)) for Dd in SETS): continue
    base = pooled(False, sel); bs = [pooled(True, sel) for _ in range(BDRAW)]
    for k_ in base: v = np.array([b_[k_][0] for b_ in bs]); rows.append(dict(group=label, item=k_, precision=base[k_][0], lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5), called=base[k_][1], true=base[k_][2]))
    for nm in ("pan_match", "lg_match"): d = np.array([b_[nm][0] - b_["afm"][0] for b_ in bs]); rows.append(dict(group=label, item=nm + "_minus_afm", precision=base[nm][0] - base["afm"][0], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    d = np.array([b_["pan_match"][0] - b_["lg_match"][0] for b_ in bs]); rows.append(dict(group=label, item="pan_minus_lg_match", precision=base["pan_match"][0] - base["lg_match"][0], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
pd.DataFrame(rows).round(4).to_csv("analysis_out/newsp/newsp_precision.csv", index=False); pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(3).to_string(index=False)); print(pd.DataFrame(nt_rows).pivot(index="species", columns="tool", values="aupr_reweighted").round(3).to_string())
