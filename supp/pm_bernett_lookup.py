#!/usr/bin/env python3
"""ProteomeLM-PPI scores of the Bernett test pairs by LOOKUP in the finished human all-vs-all file (scores.f64, pair index k = i*N - i(i+1)/2 + (j-i-1), ids sorted as in analysis_out/humanaf_allpairs/ids.txt).
Pairs with a protein outside the 19,408-protein human screen universe get NaN (counted and reported). Output analysis_out/bernett/proteomelm_bernett_test.csv (query,text,label,score). CPU only."""
import os, sys
import numpy as np, pandas as pd
D = "analysis_out/humanaf_proteomelm_universe"; OUT = "analysis_out/bernett/proteomelm_bernett_test.csv"
assert not os.path.exists(OUT), OUT + " exists, refusing"
ids = [l.strip() for l in open("analysis_out/humanaf_allpairs/ids.txt") if l.strip()]; N = len(ids); NT = N * (N - 1) // 2; ix = {p: n for n, p in enumerate(ids)}
assert os.path.exists(f"{D}/timing.json"), "ProteomeLM universe not finished"
P = np.fromfile(f"{D}/scores.f64", dtype=np.float64); assert len(P) == NT, (len(P), NT)
T = pd.read_csv("data/bernett/data/test/bernett_pairs.csv", dtype=str); T["label"] = T.label.astype(int)
a = T["query"].map(ix); b = T["text"].map(ix); ok = a.notna() & b.notna() & (T["query"] != T["text"])
i = np.minimum(a[ok].astype(np.int64), b[ok].astype(np.int64)).to_numpy(); j = np.maximum(a[ok].astype(np.int64), b[ok].astype(np.int64)).to_numpy(); k = i * N - i * (i + 1) // 2 + (j - i - 1)
T["score"] = np.nan; T.loc[ok, "score"] = P[k]
print(f"Bernett test pairs {len(T):,}; scored {int(ok.sum()):,}; missing {int((~ok).sum()):,} (positives missing {int(T.label[~ok].sum())}); proteins not in the universe: {len((set(T['query']) | set(T['text'])) - set(ix))}")
T.to_csv(OUT, index=False); print("wrote", OUT)
