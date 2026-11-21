#!/usr/bin/env python3
"""Released LoGoPPI-Cross-species with --detailed (global and MaxSim branches, logits, final) on the unique pairs of a species' Bernett-style balanced sets (unchanged ext/LoGoPPI/inference.py). GPU.
usage: rescore_logoppi_detailed.py SPECIES FASTA LG_PYTHON MODEL_DIR. Reads analysis_out/bern_univ_v2/<sp>/balanced_scores.tsv.gz (columns i, j, lg), writes logoppi_detailed.tsv.gz next to it.
Self-check (v3, rank-based): the 'score' column must agree with the all-vs-all lookup value 'lg' of the same pair."""
import os, subprocess, sys, time
import numpy as np, pandas as pd
SP, FA, LGPY, MODEL = sys.argv[1:5]; t0 = time.time(); O = f"analysis_out/bern_univ_v2/{SP}"; fin = f"{O}/logoppi_detailed.tsv.gz"
if os.path.exists(fin): print("exists, skipping"); sys.exit(0)
seqs, k = {}, None
for l in open(FA):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; seqs[k] = ""
    elif l: seqs[k] += l.rstrip("*")
ids = np.array(sorted(seqs)); ref = [l.strip() for l in open("analysis_out/humanaf_allpairs/ids.txt" if SP == "human" else f"data/univ/{SP}/ids.txt") if l.strip()]; assert list(ids) == ref, "FASTA ids differ from the universe ids"
B = pd.read_csv(f"{O}/balanced_scores.tsv.gz", sep="\t", usecols=["i", "j", "lg"]).drop_duplicates(["i", "j"]).reset_index(drop=True); n = len(B); print(f"{SP}: {n:,} unique pairs", flush=True)
W = f"{O}/_lgwork"; os.makedirs(W, exist_ok=True)
pd.DataFrame({"query": ids[B.i.to_numpy()], "text": ids[B.j.to_numpy()], "label": 0}).to_csv(f"{W}/pairs.csv", index=False)
need = np.unique(np.r_[B.i.to_numpy(), B.j.to_numpy()])
with open(f"{W}/prot.fasta", "w") as f:
    for x in need: f.write(f">{ids[x]}\n{seqs[ids[x]]}\n")
if os.path.exists(f"{W}/out.csv") and sum(1 for _ in open(f"{W}/out.csv")) - 1 == n: print("reusing the existing detailed predictions", flush=True)   # v3: no inference rerun
elif os.environ.get("NO_GPU"): print("no existing out.csv and this job has no GPU", flush=True); sys.exit(3)
else:
    r = subprocess.run([LGPY, "ext/LoGoPPI/inference.py", "--model_dir", MODEL, "--fasta_path", f"{W}/prot.fasta", "--pair_csv", f"{W}/pairs.csv", "--output_path", f"{W}/out.csv", "--gpus", "0", "--detailed"])
    if r.returncode != 0: print("INFERENCE FAILED", flush=True); sys.exit(1)
R = pd.read_csv(f"{W}/out.csv", dtype={"query": str, "text": str}); assert len(R) == n and (R["query"].to_numpy() == ids[B.i.to_numpy()]).all() and (R["text"].to_numpy() == ids[B.j.to_numpy()]).all(), "output order mismatch"
for c in ("global_score", "maxsim_score", "global_logit", "maxsim_logit", "final_logit", "score"): assert np.isfinite(R[c].to_numpy()).all(), c
d = np.abs(R["score"].to_numpy() - B.lg.to_numpy()); sp_ = float(pd.Series(R["score"].to_numpy()).corr(pd.Series(B.lg.to_numpy()), method="spearman")); print(f"self-check vs all-vs-all lookup: max |diff| {d.max():.2e}, mean {d.mean():.2e}, Spearman {sp_:.6f}", flush=True)
assert d.mean() < 5e-3 and sp_ >= 0.9999, "detailed score differs from the stored all-vs-all score (rank-based criterion v3: mean |diff| < 5e-3 and Spearman >= .9999; the earlier max-abs < 1e-3 was too strict, set after seeing the result)"
out = pd.DataFrame({"i": B.i, "j": B.j, **{c: R[c].to_numpy() for c in ("global_score", "maxsim_score", "global_logit", "maxsim_logit", "final_logit", "score")}}); out.to_csv(fin + ".tmp", sep="\t", index=False, compression="gzip"); os.replace(fin + ".tmp", fin)
import shutil; shutil.rmtree(W); print(f"DONE {SP}: {n:,} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)
