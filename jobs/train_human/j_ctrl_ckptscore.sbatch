#!/bin/bash
#SBATCH --job-name=ctrlck
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# Sensitivity analysis: score CTRL (seed 141) epoch_05.pt and epoch_08.pt with main_ctrl.py's own final-report procedure on ONE A5000 (24 GB, supports bf16 like training; same QOS/partition as t_probe_a5k).
# epoch_05.pt is the VALIDATION GATE: it must reproduce the original final report (human .9410, yeast .7275, ecoli .7685, ...) within +-0.002 AUPR, or epoch-08 numbers are not interpreted.
# A smoke pass (tiny subsets, yeast only) runs first in this same allocation.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python
CK=runs_ft/ctrl_lg_650M_800_s141/output/checkpoints
OUT=analysis_out/ctrl_ckpt_sensitivity
date; hostname; nvidia-smi -L || true
md5sum main_ctrl.py threebranch_e3.py analysis/score_ckpt_ctrl.py
ls -la $CK/epoch_05.pt $CK/epoch_08.pt || { echo "missing checkpoint"; exit 1; }
echo "===== smoke (tiny subsets, yeast only; NOT results)"
$PY -I analysis/score_ckpt_ctrl.py --ckpt $CK/epoch_05.pt --out-dir $OUT --smoke || { echo "SMOKE FAILED"; exit 1; }
echo "===== full: epoch_05 (validation gate) then epoch_08"
$PY -I analysis/score_ckpt_ctrl.py --ckpt $CK/epoch_05.pt,$CK/epoch_08.pt --out-dir $OUT
rc=$?
echo "exit $rc"
exit $rc
