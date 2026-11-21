#!/bin/bash
#SBATCH --job-name=bernlgd2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0-7%4
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}
if [ $SP = human ]; then FA=data/humanaf/screen_proteins.fasta; else FA=data/univ/$SP/proteins.fasta; fi
date; hostname; nvidia-smi -L || true; md5sum analysis/rescore_logoppi_detailed2.py $FA
./ppi_env/bin/python -u analysis/rescore_logoppi_detailed2.py $SP $FA ./lg_env/bin/python ext/LoGoPPI-Cross-species
rc=$?; echo "exit $rc"; exit $rc
