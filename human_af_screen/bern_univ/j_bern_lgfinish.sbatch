#!/bin/bash
#SBATCH --job-name=bernlgfin
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --array=0-7
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
# CPU FINISH of the LoGoPPI detailed rescoring: reuses the predictions already in <sp>/_lgwork/out.csv (exit 3 if missing), rank-based self-check.
set -uo pipefail
export NO_GPU=1 HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}
if [ $SP = human ]; then FA=data/humanaf/screen_proteins.fasta; else FA=data/univ/$SP/proteins.fasta; fi
date; hostname; nvidia-smi -L || true; md5sum analysis/rescore_logoppi_detailed3.py $FA
./ppi_env/bin/python -u analysis/rescore_logoppi_detailed3.py $SP $FA ./lg_env/bin/python ext/LoGoPPI-Cross-species
rc=$?; echo "exit $rc"; exit $rc
