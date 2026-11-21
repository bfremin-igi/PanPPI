#!/usr/bin/env python3
"""Released LoGoPPI-Cross-species on ALL pairs i<j of a protein set, one BLOCK PAIR per task (unchanged ext/LoGoPPI/inference.py). Fixed output order so scores can be re-indexed:
 p == q: np.triu_indices(n, 1) within the block; p < q: row-major (block p proteins x block q proteins). Proteins = sorted accessions of the FASTA (same order as score_allpairs ids.txt).
usage: run_logoppi_universe.py FASTA OUT_DIR LG_PYTHON MODEL_DIR TASK_ID [--blocks 8]"""
import argparse, os, subprocess, sys, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("fasta"); ap.add_argument("out"); ap.add_argument("lgpy"); ap.add_argument("model"); ap.add_argument("task", type=int); ap.add_argument("--blocks", type=int, default=8); a = ap.parse_args()
seqs, k = {}, None
for l in open(a.fasta):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; seqs[k] = ""
    elif l: seqs[k] += l.rstrip("*")
ids = np.array(sorted(seqs)); N = len(ids); K = a.blocks; bounds = [N * b // K for b in range(K + 1)]
tasks = [(p, q) for p in range(K) for q in range(p, K)]; p, q = tasks[a.task]; print(f"task {a.task} = block pair ({p},{q}); proteins {N}; blocks {K}", flush=True)
os.makedirs(a.out, exist_ok=True); fin = f"{a.out}/bp_{p}_{q}.npy"
if os.path.exists(fin): print("exists, skipping", flush=True); sys.exit(0)
Ip = np.arange(bounds[p], bounds[p + 1]); Iq = np.arange(bounds[q], bounds[q + 1])
if p == q: u, v = np.triu_indices(len(Ip), 1); i, j = Ip[u], Ip[v]
else: i = np.repeat(Ip, len(Iq)); j = np.tile(Iq, len(Ip))
n = len(i); print(f"pairs {n:,}", flush=True); work = f"{a.out}/_work_{p}_{q}"; os.makedirs(work, exist_ok=True); t0 = time.time()
pd.DataFrame({"query": ids[i], "text": ids[j], "label": 0}).to_csv(f"{work}/pairs.csv", index=False)
with open(f"{work}/prot.fasta", "w") as f:
    for x in np.unique(np.concatenate([Ip, Iq])): f.write(f">{ids[x]}\n{seqs[ids[x]]}\n")
t1 = time.time(); r = subprocess.run([a.lgpy, "ext/LoGoPPI/inference.py", "--model_dir", a.model, "--fasta_path", f"{work}/prot.fasta", "--pair_csv", f"{work}/pairs.csv", "--output_path", f"{work}/out.csv", "--gpus", "0"])
if r.returncode != 0: print("INFERENCE FAILED", flush=True); sys.exit(1)
res = pd.read_csv(f"{work}/out.csv", dtype={"query": str, "text": str}); assert len(res) == n and (res["query"].to_numpy() == ids[i]).all() and (res["text"].to_numpy() == ids[j]).all(), "output order mismatch"
s = res["score"].to_numpy().astype(np.float32); assert np.isfinite(s).all(); np.save(fin + ".tmp.npy", s); os.replace(fin + ".tmp.npy", fin)
import shutil; shutil.rmtree(work); print(f"DONE task {a.task} ({p},{q}): {n:,} pairs in {(time.time() - t1) / 60:.1f} min inference = {n / (time.time() - t1):,.0f} pairs/s; total {(time.time() - t0) / 60:.1f} min", flush=True)
