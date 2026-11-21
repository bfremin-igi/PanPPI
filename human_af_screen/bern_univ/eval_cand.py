import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU, average_precision_score as AP
SEEDS = ("s42", "s7", "s141"); SPS = sys.argv[1:] or ["human", "ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"]; rows = []; ov = []; drops = []
for sp in SPS:
    d = f"out3/{sp}"
    if not os.path.exists(f"{d}/cand_scores.tsv.gz"): print("missing", sp); continue
    B = pd.read_csv(f"{d}/cand_scores.tsv.gz", sep="\t"); meta = {(m["tier"], m["level"], m["variant"]): m for m in json.load(open(f"{d}/cand_meta.json")) if "note" not in m}
    PM = pd.read_csv(f"{d}/pm_scores.tsv.gz", sep="\t"); B["_a"] = np.minimum(B.i, B.j); B["_b"] = np.maximum(B.i, B.j); B = B.merge(PM.rename(columns={"i": "_a", "j": "_b"}), on=["_a", "_b"], how="left"); assert B.pm.notna().all(), sp
    for s in SEEDS: B[f"bl_{s}"] = 0.5 * (B[f"g_{s}"] + B[f"m_{s}"])
    B["pan"] = B[[f"bl_{s}" for s in SEEDS]].mean(axis=1)
    for (tier, lvl, var), m in meta.items():
        G = B[(B.tier == tier) & (B.level == lvl) & (B.variant == var)]; P = G[G.label == 1]; reps = sorted(set(G[G.label == 0].rep)); frac = min(m["neg_per_rep"]) / m["n_pos"]; dg = []; res = {}; ovs = []; lgq = []
        for r in reps:
            N = G[(G.label == 0) & (G.rep == r)]; k = min(len(N), len(P)); Ps = P.sample(k, random_state=r + 1) if len(P) > k else P; D = pd.concat([Ps, N.sample(k, random_state=r + 7) if len(N) > k else N]); y = D.label.to_numpy()
            c = pd.concat([D.i, D.j]).value_counts(); dg.append(AU(y, np.log1p(D.i.map(c).to_numpy()) + np.log1p(D.j.map(c).to_numpy())))
            T = {"PanPPI blend": [D[f"bl_{s}"] for s in SEEDS], "PanPPI global": [D[f"g_{s}"] for s in SEEDS], "LoGoPPI": [D.lg], "ProteomeLM": [D.pm], "AFM ipSAE": [D.ipsae], "AFM ipTM": [D.iptm]}
            for nm, vs in T.items(): res.setdefault(nm, []).append((np.mean([AU(y, v) for v in vs]), np.mean([AP(y, v) for v in vs])))
            res.setdefault("AFM confident flag", []).append((AU(y, D.afm_pass), AP(y, D.afm_pass)))
            thr = np.quantile(D[D.label == 0].pan, 0.95); call = (D.pan > thr).to_numpy(); afm = D.afm_pass.to_numpy().astype(bool)
            thl = np.quantile(D[D.label == 0].lg, 0.95); lcall = (D.lg > thl).to_numpy()
            lgq.append([float(y[(lcall == a_) & (afm == b_)].mean()) if ((lcall == a_) & (afm == b_)).sum() else np.nan for a_ in (True, False) for b_ in (True, False)] + [int(((lcall == a_) & (afm == b_)).sum()) for a_ in (True, False) for b_ in (True, False)])
            ovs.append((float(pd.Series(D.pan.to_numpy()).corr(pd.Series(D.ipsae.to_numpy()), method="spearman")), float(pd.Series(D.pan.to_numpy()).corr(pd.Series(D.iptm.to_numpy()), method="spearman")),
                        float(pd.Series(D.pan.to_numpy()[y == 1]).corr(pd.Series(D.ipsae.to_numpy()[y == 1]), method="spearman")), *[float(y[(call == a) & (afm == b)].mean()) if ((call == a) & (afm == b)).sum() else np.nan for a in (True, False) for b in (True, False)], *[int(((call == a) & (afm == b)).sum()) for a in (True, False) for b in (True, False)], float(call[y == 1].mean()), float(afm[y == 1].mean())))
        deg = float(np.mean(dg)); ok = frac >= 0.8 and deg <= 0.60
        if not ok: drops.append((sp, tier, lvl, var, m["n_pos"], round(frac, 2), round(deg, 3)))
        out = {"species": sp, "tier": tier, "level": lvl, "variant": var, "n_pos": m["n_pos"], "neg_frac": round(frac, 2), "deg_only_auroc": round(deg, 3), "kept": ok}
        for nm, v in res.items(): out[nm + " auroc"] = np.mean([x[0] for x in v]); out[nm + " aupr"] = np.mean([x[1] for x in v])
        rows.append(out); o = np.nanmean(np.array(ovs), axis=0); ov.append({"species": sp, "tier": tier, "level": lvl, "variant": var, "n_pos": m["n_pos"], "kept": ok, "rho_pan_ipsae_all": o[0], "rho_pan_iptm_all": o[1], "rho_pan_ipsae_pos": o[2], "truth_rate_both": o[3], "truth_rate_panonly": o[4], "truth_rate_afmonly": o[5], "truth_rate_neither": o[6], "n_both": o[7], "n_panonly": o[8], "n_afmonly": o[9], "n_neither": o[10], "recall_pan_call": o[11], "recall_afm_call": o[12], **{f"lg_{k}": v for k, v in zip(["rate_both", "rate_lgonly", "rate_afmonly", "rate_neither", "n_both", "n_lgonly", "n_afmonly", "n_neither"], np.nanmean(np.array(lgq), axis=0))}})
R = pd.DataFrame(rows); R.round(4).to_csv("cand_results.csv", index=False); O = pd.DataFrame(ov); O.round(4).to_csv("cand_overlap.csv", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for var in ("A", "B"):
    K = R[(R.variant == var) & R.kept & (R.level != "L1")]
    print(f"\nVARIANT {var} (negatives: {'all low-STRING candidates' if var == 'A' else 'low-STRING candidates without AFM-confident pairs, favours AFM'}); kept cells {len(K)}; AUPR")
    print(K[["species", "tier", "level", "n_pos", "deg_only_auroc", "PanPPI blend aupr", "PanPPI global aupr", "LoGoPPI aupr", "AFM ipSAE aupr", "AFM ipTM aupr", "AFM confident flag aupr"]].round(3).to_string(index=False))
    print("PanPPI blend > AFM ipSAE:", int((K["PanPPI blend aupr"] > K["AFM ipSAE aupr"]).sum()), "of", len(K), "| > ipTM:", int((K["PanPPI blend aupr"] > K["AFM ipTM aupr"]).sum()), "| > AFM flag:", int((K["PanPPI blend aupr"] > K["AFM confident flag aupr"]).sum()), "| > LoGoPPI:", int((K["PanPPI blend aupr"] > K["LoGoPPI aupr"]).sum()))
print("\nDROPPED:", len(drops), drops[:30])
