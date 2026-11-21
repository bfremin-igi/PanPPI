#!/bin/bash
#SBATCH --job-name=posthoc
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# POST-HOC UNCERTAINTY for a finished plain-head run (E3a by default). BUILT 2026-11-12, NOT SUBMITTED, NOT APPROVED TO RUN.
# It needs the finished run's calibration.json and best.pt, so submit it only after E3a (41254062) completes:
#   DATA=data/logoppi/data sbatch --export=ALL --dependency=afterok:41254062 jobs/j_posthoc_e3a.sbatch
# Same small-card choice as j_db55 / j_e3diag and for the same reason: batch-16 no-grad inference with ESM-2 650M in fp32 (GTX2080TI is
# Turing, no bf16), and it must not contend with anything on the H200 node. Reads the run, writes ONLY under <run>/posthoc_gp.
# Memory: the ~56k training proteins are embedded POOLED-ONLY (embed_pooled), because a full fp16 residue cache for them would be ~23 GB.
# Time: embedding ~90k proteins in fp32 on a 2080Ti is UNMEASURED (my estimate ~1-2 h); the wall is 6 h. Estimates here have run
# optimistic five times -- read the progress line before trusting it.
# Readouts U0-U5 are fixed in analysis/posthoc_uncertainty.py's docstring before any number exists.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python
DATA="${DATA:?set DATA to the LoGoPPI bundle data/ dir}"
RUN=${RUN:-runs_ft/e3a_nounc_650M_800_s141}
date; nvidia-smi -L || true
test -f $RUN/calibration.json && test -f $RUN/output/checkpoints/best.pt || { echo "RUN $RUN is not finished (no calibration.json / best.pt)"; exit 1; }
echo "===== smoke: post-hoc uncertainty unit tests (16 checks, CPU)"
$PY -I tests/test_posthoc_uncertainty.py || { echo "SMOKE FAILED"; exit 1; }
echo "===== post-hoc GP on $RUN"
$PY -I analysis/posthoc_uncertainty.py --run-dir $RUN --esm-dir ext/esm2_650M_hf --data-dir $DATA --dtype fp32
echo "exit $?"
