#!/bin/bash
#SBATCH --job-name=nsp2cell
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>; PY=./ppi_env/bin/python; SPS=(soybean arabidopsis rat rice zebrafish); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; date; hostname; md5sum analysis/newsp_cells.py data/newsp/$SP/tiers.tsv.gz
$PY -u analysis/newsp_cells.py $SP; rc=$?; echo "exit $rc"; exit $rc
