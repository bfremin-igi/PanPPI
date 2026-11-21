#!/bin/bash
#SBATCH --job-name=uncbern
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SEEDS=(42 7 141); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; PY=./ppi_env/bin/python; RUN=runs_ft/e3a_bernett_650M_800_s${S}_h1; OUT=analysis_out/uncert_bernett_s$S
[ -e $OUT ] && { echo "$OUT exists, refusing"; echo "exit 1"; exit 1; }
test -f $RUN/output/checkpoints/best.pt || { echo "no checkpoint in $RUN"; echo "exit 1"; exit 1; }
date; hostname; nvidia-smi -L || true; md5sum analysis/uncert_features_bernett.py $RUN/output/checkpoints/best.pt
echo "===== smoke (NOT results): 3,000 training proteins and 3,000 pairs per set"
rm -rf analysis_out/_uncbern_smoke_$S
$PY -I analysis/uncert_features_bernett.py $RUN analysis_out/_uncbern_smoke_$S --limit 3000 --T 4 || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf analysis_out/_uncbern_smoke_$S
echo "===== full run"
$PY -I analysis/uncert_features_bernett.py $RUN $OUT --T 8
rc=$?; echo "exit $rc"; exit $rc
