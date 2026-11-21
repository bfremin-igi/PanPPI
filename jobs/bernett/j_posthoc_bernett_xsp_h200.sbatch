#!/bin/bash
#SBATCH --job-name=phbx
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
# Same script and readouts U0-U5 as the cross-species post-hoc run (analysis/posthoc_uncertainty_chunked.py, leverage RFF-GP on the plain head's penultimate features, fitted AFTER training on the Bernett TRAIN pairs).
# Reads the run, writes ONLY under $RUN/posthoc_bernett_xsp. bf16 (matches training autocast; H200 has it). Smoke first (unit tests + 200-pair GPU end-to-end), then the full set. Propagates the exit code.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SEEDS=(42 7 141); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; RUN=runs_ft/e3a_bernett_650M_800_s${S}_h1; DATA=data/bernett_xsp_combo; PY=./ppi_env/bin/python
date; hostname; nvidia-smi -L || true; md5sum analysis/posthoc_uncertainty_chunked.py $RUN/output/checkpoints/best.pt $RUN/calibration.json
# data dir = Bernett train/validation files + the Bernett-style cross-species test files, assembled with symlinks (new directory, no copy)
mkdir -p data/bernett_xsp_combo && for f in human_train_clean.csv human_validation_selection.csv human_validation_calibration.csv proteins.fasta; do ln -sfn ../bernett/data/$f data/bernett_xsp_combo/$f; done; ln -sfn ../xsp_bern/data/test data/bernett_xsp_combo/test
test -f $RUN/calibration.json && test -f $RUN/output/checkpoints/best.pt || { echo "RUN $RUN not finished"; echo "exit 1"; exit 1; }
test -s $RUN/posthoc_bernett_xsp/summary.json && { echo "output exists"; exit 0; }
echo "===== smoke 1: unit tests (CPU)"; $PY -I tests/test_posthoc_uncertainty.py || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
echo "===== smoke 2: GPU end-to-end, 200 pairs, throw-away dir (NOT results)"
SM=$RUN/_posthoc_bx_smoke; rm -rf $SM
$PY -I analysis/posthoc_uncertainty_chunked.py --run-dir $RUN --esm-dir ext/esm2_650M_hf --data-dir $DATA --dtype bf16 --species yeast,mouse,fly,worm,ecoli --eval-subset 200 --out-dir $SM || { echo "GPU SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf $SM
echo "===== post-hoc GP on $RUN, Bernett test"
$PY -I analysis/posthoc_uncertainty_chunked.py --run-dir $RUN --esm-dir ext/esm2_650M_hf --data-dir $DATA --dtype bf16 --species yeast,mouse,fly,worm,ecoli --out-dir $RUN/posthoc_bernett_xsp
rc=$?; echo "exit $rc"; exit $rc
