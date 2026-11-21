#!/bin/bash
#SBATCH --job-name=nsp2hom
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>; SPS=(soybean arabidopsis rat rice zebrafish); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; date; hostname
bash analysis/newsp_homology.sh $SP; rc=$?; echo "exit $rc"; exit $rc
