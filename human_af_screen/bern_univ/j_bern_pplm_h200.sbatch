#!/bin/bash
#SBATCH --job-name=bernpplmh
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python; T=$SLURM_ARRAY_TASK_ID; B=ext/TUnA/data/processed/bernett; OUT=analysis_out/bernett_pplm; D=data/xsp_matched/bernett_pplm/s$T
mkdir -p $OUT $D logs
test -s $OUT/s$T.csv && { echo "output exists"; exit 0; }
date; hostname; nvidia-smi -L || true; md5sum $B/Intra2_interaction.tsv $B/Intra2_dictionary.tsv analysis/zeroshot_pplm.py
awk -v t=$T 'NR % 4 == t' $B/Intra2_interaction.tsv > $D/test_interaction.tsv
awk -v t=$T 'BEGIN{OFS="\t"; print "rec_id","pep_id","label"} NR % 4 == t {print $1,$2,$3}' $B/Intra2_interaction.tsv > $D/test_meta.tsv
echo "shard $T pairs $(wc -l < $D/test_interaction.tsv)"
echo "===== smoke: 64 pairs (NOT results)"
W=$OUT/_smoke$T; rm -rf $W; mkdir -p $W
$PY -u analysis/zeroshot_pplm.py --ppl-dir ext/PPLM --weights-dir ext/weights/pplm --data-dir $D --dictionary $B/Intra2_dictionary.tsv --out-dir $W --parts test --limit 64 || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf $W
echo "===== full"
W=$OUT/_work$T; mkdir -p $W
$PY -u analysis/zeroshot_pplm.py --ppl-dir ext/PPLM --weights-dir ext/weights/pplm --data-dir $D --dictionary $B/Intra2_dictionary.tsv --out-dir $W --parts test || { echo "SCORING FAILED (work dir kept for resume)"; echo "exit 1"; exit 1; }
$PY -I - $W $D/test_interaction.tsv $OUT/s$T.csv <<'PYEOF' || { echo "COLLECT FAILED"; echo "exit 1"; exit 1; }
import sys, os, pandas as pd
W, inter, out = sys.argv[1:4]
r = pd.read_csv(f"{W}/evaluation_results_test.tsv", sep="\t")
p = pd.read_csv(inter, sep="\t", header=None, names=["query", "text", "label"], dtype=str)
assert len(r) == len(p), ("row count", len(r), len(p))
assert (r["Protein A"].astype(str).values == p["query"].values).all() and (r["Protein B"].astype(str).values == p["text"].values).all(), "row order differs from the pair file"
assert r["S"].notna().all(), "NaN scores"
pd.DataFrame({"query": p["query"], "text": p["text"], "label": p["label"], "score": r["S"]}).to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out); print("collected", len(r), "rows ->", out)
PYEOF
rm -rf $W
echo "exit 0"
