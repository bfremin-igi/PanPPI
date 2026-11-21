"""Compare the panppi tool's output with the scores stored by the original run (same model, same pairs). usage: tool_check_compare.py TOOL_CSV STORED_TSV [MAX_TOL=0.1] [MEAN_TOL=0.005] [AUPR_TOL=0.0005]
Pass = every pair matched, AUPR equal within AUPR_TOL, mean |blend diff| < MEAN_TOL and max < MAX_TOL (bf16 padding changes the last digits, so exact equality is not expected)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
t = pd.read_csv(sys.argv[1]); s = pd.read_csv(sys.argv[2], sep="\t", usecols=["a", "b", "label", "cal_global", "cal_maxsim", "cal_logoppi_g+m"])
m = t.merge(s, left_on=["protein_a", "protein_b"], right_on=["a", "b"], how="inner")
print(f"pairs in tool output {len(t)}, matched in stored file {len(m)}")
dg = (m.global_logit - m.cal_global).abs(); dm = (m.maxsim_logit - m.cal_maxsim).abs(); db = (m.blend_logit - m["cal_logoppi_g+m"]).abs()
print(f"max |global diff| {dg.max():.2e}  max |maxsim diff| {dm.max():.2e}  max |blend diff| {db.max():.2e}  (mean {db.mean():.2e})")
print(f"AUPR tool {AP(m.label, m.blend_logit):.4f}  stored {AP(m.label, m['cal_logoppi_g+m']):.4f}")
mx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1; mn = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005; at = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0005
ok = len(m) == len(t) and db.max() < mx and db.mean() < mn and abs(AP(m.label, m.blend_logit) - AP(m.label, m["cal_logoppi_g+m"])) < at
print("COMPARE OK" if ok else "COMPARE FAILED", f"(criteria: max<{mx}, mean<{mn}, AUPR within {at})")
