#!/bin/bash
#SBATCH --job-name=bernpm2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --time=00:30:00
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}
date; hostname; md5sum analysis/add_pm_lookup2.py
test -s analysis_out/bern_univ_v2/$SP/pm_scores.tsv.gz && { echo "output exists"; exit 0; }
./ppi_env/bin/python -u analysis/add_pm_lookup2.py $SP
rc=$?; echo "exit $rc"; exit $rc
