#!/bin/bash
#SBATCH --job-name=candneg2f
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; O=analysis_out/bern_cand2/$SP
date; hostname; md5sum analysis/build_cand_negs2.py
test -s $O/cand_scores.tsv.gz && { echo "output exists"; exit 0; }
./ppi_env/bin/python -u analysis/build_cand_negs2.py $SP
rc=$?; echo "exit $rc"; exit $rc
