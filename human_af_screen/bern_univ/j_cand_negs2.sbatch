#!/bin/bash
#SBATCH --job-name=candneg2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --time=04:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; O=analysis_out/bern_cand2/$SP
date; hostname; md5sum analysis/build_cand_negs2.py
test -s $O/cand_scores.tsv.gz && { echo "output exists"; exit 0; }
./ppi_env/bin/python -u analysis/build_cand_negs2.py $SP
rc=$?; echo "exit $rc"; exit $rc
