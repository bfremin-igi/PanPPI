#!/bin/bash
#SBATCH --job-name=pmbern
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
cd <SCRATCH>
date; hostname; ./ppi_env/bin/python -I analysis/pm_bernett_lookup.py
rc=$?; echo "exit $rc"; exit $rc
