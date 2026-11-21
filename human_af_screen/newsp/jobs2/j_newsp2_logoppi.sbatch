#!/bin/bash
#SBATCH --job-name=nsp2lg
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>; SPS=(soybean arabidopsis rat rice zebrafish); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; date; hostname; nvidia-smi -L || true; wc -l analysis_out/newsp/$SP/cell_pairs.csv
bash analysis/newsp_cells_logoppi.sh $SP; rc=$?; echo "exit $rc"; exit $rc
