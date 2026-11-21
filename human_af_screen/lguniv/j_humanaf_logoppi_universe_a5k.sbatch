#!/bin/bash
#SBATCH --job-name=lguniv
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0-35%4
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python; LG=./lg_env/bin/python; M=ext/LoGoPPI-Cross-species; FA=data/humanaf/screen_proteins.fasta; OUT=analysis_out/humanaf_logoppi_universe
date; hostname; nvidia-smi -L || true; md5sum analysis/run_logoppi_universe.py $FA
$PY -u analysis/run_logoppi_universe.py $FA $OUT $LG $M $SLURM_ARRAY_TASK_ID --blocks 8
rc=$?; echo "exit $rc"; exit $rc
