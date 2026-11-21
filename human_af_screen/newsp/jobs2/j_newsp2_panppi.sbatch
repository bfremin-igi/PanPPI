#!/bin/bash
#SBATCH --job-name=nsp2pan
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --array=0-14
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>; export PYTHONPATH=$PWD/panppi_tool; PY=./ppi_env/bin/python
SPS=(soybean arabidopsis rat rice zebrafish); TAGS=(s42 s7 s141); RUNS=(runs_ft/e3a_nounc_650M_800_s42_r2 runs_ft/e3a_nounc_650M_800_s7 runs_ft/e3a_nounc_650M_800_s141_clean2)
SP=${SPS[$((SLURM_ARRAY_TASK_ID / 3))]}; K=$((SLURM_ARRAY_TASK_ID % 3)); date; hostname; nvidia-smi -L || true; md5sum analysis/newsp_cells_panppi.py analysis/newsp/$SP/cell_pairs.csv 2>/dev/null; wc -l analysis_out/newsp/$SP/cell_pairs.csv
echo "===== smoke (NOT results)"; $PY -u analysis/newsp_cells_panppi.py $SP ${RUNS[$K]} ${TAGS[$K]} --smoke || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
echo "===== full"; $PY -u analysis/newsp_cells_panppi.py $SP ${RUNS[$K]} ${TAGS[$K]}; rc=$?; echo "exit $rc"; exit $rc
