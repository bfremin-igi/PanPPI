"""Paired protein-clustered bootstrap intervals for the standard-file uncertainty boost (B1/B2/B3 minus B0), AUPR averaged over seeds within each resample. Test-set noise only."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unc_boost import *
from boot_lib import Boot
import boot_lib; boot_lib.B = 1000
seeds = sys.argv[1:] or ["42", "141"]; rows = []; store = {}
for s in seeds:
    UD = UI + f"analysis_out/uncert_s{s}/"; train, _ = load_emb(UD + "prot_humantrain"); D = {}
    for sp in SP8: D[sp] = build(UI + f"runs_ft/{RUNS[s]}/posthoc_gp/scores_{sp}_posthoc.tsv", UD + f"pairs_{sp}.tsv.gz", UD + f"prot_{sp}", train)
    for te in CORE:
        Ftr = pd.concat([D[x][0] for x in SP8 if x != te]); ytr = np.concatenate([D[x][1] for x in SP8 if x != te]); Fte, yte, a, b = D[te]
        store.setdefault(te, {"y": yte, "a": a, "b": b, "sc": {}}); [store[te]["sc"].__setitem__(f"{n}_s{s}", fit_predict(Ftr, ytr, Fte, c)) for n, c in SETS.items()]
for te in CORE:
    st = store[te]; bt = Boot(st["a"], st["b"], st["y"])
    for n, v in st["sc"].items(): bt.prep(n, v)
    groups = {n: [f"{n}_s{s}" for s in seeds] for n in SETS}
    full, out = bt.run(groups, "B0")
    for o in out: rows.append(dict(set=te, **{k: v for k, v in o.items()}))
    print(te, " | ".join("%s %+.4f [%+.4f, %+.4f]" % (o["tool"], -o["diff"], -o["hi"], -o["lo"]) for o in out), flush=True)
pd.DataFrame(rows).round(5).to_csv(os.path.join(HERE, "unc_boost_standard_ci.csv"), index=False)
