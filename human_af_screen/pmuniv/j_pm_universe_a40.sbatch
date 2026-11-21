#!/bin/bash
#SBATCH --job-name=pmuniv
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --exclude=n0214.savio3
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd <SCRATCH>
FA=<SCRATCH>/data/humanaf/screen_proteins.fasta; IDS=<SCRATCH>/analysis_out/humanaf_allpairs/ids.txt; OUT=<SCRATCH>/analysis_out/humanaf_proteomelm_universe
date; hostname; nvidia-smi -L || true; md5sum analysis/pm_universe.py $FA $IDS
test -s $OUT/timing.json && { echo "output exists"; exit 0; }
mkdir -p $OUT; cd ext/ProteomeLM
../pm_env/bin/python -u ../../analysis/pm_universe.py $FA $IDS $OUT --chunk 4000000
rc=$?; echo "exit $rc"; exit $rc
