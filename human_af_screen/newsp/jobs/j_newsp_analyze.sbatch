#!/bin/bash
#SBATCH --job-name=newspan
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
cd <SCRATCH>; PY=./ppi_env/bin/python; date; hostname; md5sum analysis/newsp_analyze.py
$PY -u analysis/newsp_analyze.py; rc=$?; echo "exit $rc"; exit $rc
