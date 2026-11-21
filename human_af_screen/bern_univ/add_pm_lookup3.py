#!/usr/bin/env python3
"""ProteomeLM-PPI scores of the unique pairs of a species' balanced sets, by LOOKUP in the finished all-vs-all file (scores.f64, pair index k = i*N - i(i+1)/2 + (j-i-1), sorted ids). CPU.
usage: add_pm_lookup.py SPECIES  (writes analysis_out/bern_univ_v2/<sp>/pm_scores.tsv.gz with i, j, pm)."""
import os, sys
import numpy as np, pandas as pd
SP = sys.argv[1]; O = f"analysis_out/bern_univ_v2/{SP}"; D = "analysis_out/humanaf_proteomelm_universe" if SP == "human" else f"analysis_out/univ_{SP}/proteomelm"
ids = [l.strip() for l in open("analysis_out/humanaf_allpairs/ids.txt" if SP == "human" else f"data/univ/{SP}/ids.txt") if l.strip()]; N = len(ids); NT = N * (N - 1) // 2
assert os.path.exists(f"{D}/timing.json"), f"ProteomeLM universe for {SP} not finished"
P = np.fromfile(f"{D}/scores.f64", dtype=np.float64); assert len(P) == NT, (len(P), NT)   # v3: one sequential read instead of random access through a memory map
B = pd.read_csv(f"{O}/balanced_scores.tsv.gz", sep="\t", usecols=["i", "j"]).drop_duplicates(["i", "j"]).reset_index(drop=True); i = np.minimum(B.i.to_numpy(), B.j.to_numpy()).astype(np.int64); j = np.maximum(B.i.to_numpy(), B.j.to_numpy()).astype(np.int64)
k = i * N - i * (i + 1) // 2 + (j - i - 1); v = P[k]; assert np.isfinite(v).all()
pd.DataFrame({"i": i, "j": j, "pm": v}).to_csv(f"{O}/pm_scores.tsv.gz", sep="\t", index=False, compression="gzip"); print(f"{SP}: {len(B):,} pairs, wrote {O}/pm_scores.tsv.gz")
