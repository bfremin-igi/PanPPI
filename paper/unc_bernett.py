"""Uncertainty boost under the Bernett protocol and on the Bernett-style cross-species sets (PROTOCOL pre-registered in supplement_plan.md before any Bernett number existed; same features and boosts B0-B3 as unc_boost.py).
(A) Bernett: fit on the Bernett validation SELECTION half (the half that carries GP variance), evaluate on the Bernett test set, per seed.
(B) Transfer check: boost fitted on the 8 standard species (human-trained model, same seed) applied to the Bernett test features of the Bernett-trained model.
(C) Bernett-style cross-species sets: leave-one-set-out over the five sets (fit on four, test on the fifth), per seed.
Per seed AUPR/AUROC; paired protein-clustered bootstrap (1,000 resamples, seeds averaged inside each resample) for B1/B2/B3 minus B0. Output unc_bernett.csv, unc_bernett_ci.csv."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unc_boost import *
from boot_lib import Boot
import boot_lib; boot_lib.B = 1000
SEEDS = ["42", "7", "141"]; XS = ["yeast", "mouse", "fly", "worm", "ecoli"]; RB = lambda s: UI + f"runs_ft/e3a_bernett_650M_800_s{s}_h1/"; rows = []; store = {}
def keep(name, y, a, b, sc, s):
    st = store.setdefault(name, {"y": y, "a": a, "b": b, "sc": {}}); st["sc"].update({f"{n}_s{s}": v for n, v in sc.items()})
for s in SEEDS:
    UD = UI + f"analysis_out/uncert_bernett_s{s}/"; train, _ = load_emb(UD + "prot_train")
    Ft, yt, at, bt = build(RB(s) + "posthoc_bernett/scores_bernett_posthoc.tsv", UD + "pairs_bernett.tsv.gz", UD + "prot_bernett", train)
    Fv, yv, _, _ = build(RB(s) + "posthoc_bernett/scores_human_selection_posthoc.tsv", UD + "pairs_val_selection.tsv.gz", UD + "prot_val_selection", train)
    sc = {n: fit_predict(Fv, yv, Ft, c) for n, c in SETS.items()}
    for n, v in sc.items(): rows.append(dict(analysis="A_bernett_val_fit", seed=s, set="bernett", boost=n, aupr=AP(yt, v), auroc=AU(yt, v), n=len(yt)))
    keep("A_bernett_val_fit", yt, at, bt, sc, s)
    # (B) transfer from the standard species (human-trained model, same seed)
    US = UI + f"analysis_out/uncert_s{s}/"; trs, _ = load_emb(US + "prot_humantrain"); Fs = []; ys = []
    for sp in SP8: F, y, _, _ = build(UI + f"runs_ft/{RUNS[s]}/posthoc_gp/scores_{sp}_posthoc.tsv", US + f"pairs_{sp}.tsv.gz", US + f"prot_{sp}", trs); Fs.append(F); ys.append(y)
    Fs = pd.concat(Fs); ys = np.concatenate(ys); sc = {n: fit_predict(Fs, ys, Ft, c) for n, c in SETS.items()}
    for n, v in sc.items(): rows.append(dict(analysis="B_transfer_from_standard", seed=s, set="bernett", boost=n, aupr=AP(yt, v), auroc=AU(yt, v), n=len(yt)))
    keep("B_transfer_from_standard", yt, at, bt, sc, s)
    # (C) Bernett-style cross-species sets, leave-one-set-out
    UX = UI + f"analysis_out/uncert_bernett_xsp_s{s}/"; trx, _ = load_emb(UX + "prot_train"); X = {}
    for sp in XS: X[sp] = build(RB(s) + f"posthoc_bernett_xsp/scores_{sp}_posthoc.tsv", UX + f"pairs_{sp}.tsv.gz", UX + f"prot_{sp}", trx)
    for te in XS:
        Ftr = pd.concat([X[x][0] for x in XS if x != te]); ytr = np.concatenate([X[x][1] for x in XS if x != te]); Fte, yte, a, b = X[te]
        sc = {n: fit_predict(Ftr, ytr, Fte, c) for n, c in SETS.items()}
        for n, v in sc.items(): rows.append(dict(analysis="C_bernett_style_LOSO", seed=s, set=te, boost=n, aupr=AP(yte, v), auroc=AU(yte, v), n=len(yte)))
        keep("C_" + te, yte, a, b, sc, s)
    print("seed", s, "done", flush=True)
R = pd.DataFrame(rows); R.round(5).to_csv(HERE + "/unc_bernett.csv", index=False)
M = R.groupby(["analysis", "set", "boost"]).aupr.mean().unstack("boost"); print("\nAUPR, mean over seeds\n", M.round(4).to_string())
ci = []
for name, st in store.items():
    bt_ = Boot(st["a"], st["b"], st["y"])
    for n, v in st["sc"].items(): bt_.prep(n, v)
    full, out = bt_.run({n: [f"{n}_s{s}" for s in SEEDS] for n in SETS}, "B0")
    for o in out: ci.append(dict(analysis=name, **o))
    print(name, " | ".join("%s %+.4f [%+.4f, %+.4f]" % (o["tool"], -o["diff"], -o["hi"], -o["lo"]) for o in out), flush=True)
pd.DataFrame(ci).round(5).to_csv(HERE + "/unc_bernett_ci.csv", index=False)
