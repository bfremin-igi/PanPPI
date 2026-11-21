#!/bin/bash
#SBATCH --job-name=candpm
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --time=01:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
# v3: scores.f64 read into memory once (the v2 memory-mapped lookup timed out for human and mouse).
set -uo pipefail
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}
date; hostname; md5sum analysis/add_pm_cand.py
test -s analysis_out/bern_cand2/$SP/pm_scores.tsv.gz && { echo "output exists"; exit 0; }
./ppi_env/bin/python -u analysis/add_pm_cand.py $SP
rc=$?; echo "exit $rc"; exit $rc
