#!/bin/bash
#SBATCH --job-name=newsplg
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SPS=(soybean arabidopsis rat rice zebrafish); SP=${SPS[$((SLURM_ARRAY_TASK_ID / 4))]}; ST=$((SLURM_ARRAY_TASK_ID % 4)); date; hostname; nvidia-smi -L || true; md5sum analysis/newsp_logoppi.sh analysis/run_logoppi_universe.py data/newsp/$SP/needed.fasta
bash analysis/newsp_logoppi.sh $SP $ST; rc=$?; echo "exit $rc"; exit $rc
