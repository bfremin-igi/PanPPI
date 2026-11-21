#!/usr/bin/env python3
"""Lay out the Bernett-style balanced cross-species pairs in the LoGoPPI test-dir format (<set>_pairs.csv with query,text,label; <set>.fasta symlinked to the original species FASTA) plus a 2,000-pair
check set 'bchk' (first rows of the Bernett test file) under data/xsp_bern/data/test. Idempotent, new files only. usage: prep_xsp_bern.py (run from the FastPPI2 root)"""
import os, pandas as pd
D = "data/xsp_bern/data/test"; os.makedirs(D, exist_ok=True); A = pd.read_csv("data/xsp_bern/pairs_all.tsv.gz", sep="\t", dtype=str)
for sp, g in A.groupby("set", sort=False):
    out = f"{D}/{sp}_pairs.csv"
    if not os.path.exists(out): g[["query", "text", "label"]].to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out)
    fa = f"{D}/{sp}.fasta"
    if not os.path.lexists(fa): os.symlink(os.path.abspath(f"data/logoppi/data/test/{sp}.fasta"), fa)
    print(sp, len(g), "pairs", flush=True)
B = "ext/LoGoPPI-Bernett/data/bernett"; t = pd.read_csv(f"{B}/test.csv", dtype=str).iloc[:2000]
if not os.path.exists(f"{D}/bchk_pairs.csv"): t[["query", "text", "label"]].to_csv(f"{D}/bchk_pairs.csv.tmp", index=False); os.replace(f"{D}/bchk_pairs.csv.tmp", f"{D}/bchk_pairs.csv")
if not os.path.lexists(f"{D}/bchk.fasta"): os.symlink(os.path.abspath(f"{B}/proteins.fasta"), f"{D}/bchk.fasta")
print("bchk 2000 pairs")
