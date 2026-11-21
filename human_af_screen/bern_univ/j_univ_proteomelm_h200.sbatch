#!/bin/bash
#SBATCH --job-name=univpmh
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd <SCRATCH>
SPS=(ecoli mtb paer yeast fly); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; R=<SCRATCH>
FA=$R/data/univ/$SP/proteins.fasta; IDS=$R/data/univ/$SP/ids.txt; OUT=$R/analysis_out/univ_$SP/proteomelm
date; hostname; nvidia-smi -L || true; echo "task $SLURM_ARRAY_TASK_ID: $SP"; md5sum analysis/pm_universe.py $FA $IDS
test -s $OUT/timing.json && { echo "output exists"; exit 0; }
mkdir -p $OUT; cd ext/ProteomeLM
../pm_env/bin/python -u ../../analysis/pm_universe.py $FA $IDS $OUT --chunk 4000000
rc=$?; echo "exit $rc"; exit $rc
