#!/bin/bash
#SBATCH --job-name=unc7
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python; RUN=runs_ft/e3a_nounc_650M_800_s7; OUT=analysis_out/uncert_s7
[ -e $OUT ] && { echo "$OUT exists, refusing"; echo "exit 1"; exit 1; }
test -f $RUN/output/checkpoints/best.pt || { echo "no checkpoint in $RUN"; echo "exit 1"; exit 1; }
date; hostname; nvidia-smi -L || true; md5sum analysis/uncert_features_std.py $RUN/output/checkpoints/best.pt
echo "===== smoke (NOT results): 3,000 human training proteins and 3,000 pairs per set"
rm -rf analysis_out/_unc7_smoke
$PY -I analysis/uncert_features_std.py $RUN analysis_out/_unc7_smoke --limit 3000 --T 4 || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf analysis_out/_unc7_smoke
echo "===== full run"
$PY -I analysis/uncert_features_std.py $RUN $OUT --T 8
rc=$?; echo "exit $rc"; exit $rc
