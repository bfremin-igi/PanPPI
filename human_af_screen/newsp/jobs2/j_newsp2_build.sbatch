#!/bin/bash
#SBATCH --job-name=nsp2build
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
cd <SCRATCH>; PY=./ppi_env/bin/python; date; hostname; md5sum analysis/newsp_build_inputs.py
$PY -u analysis/newsp_build_inputs.py; rc=$?; echo "exit $rc"; exit $rc
