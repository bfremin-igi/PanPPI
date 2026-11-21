#!/bin/bash
#SBATCH --job-name=benchpm
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd <SCRATCH>
IN=<SCRATCH>/analysis_out/bench_h200_4runs/inputs; OUT=<SCRATCH>/analysis_out/bench_proteomelm_h200
date; hostname; nvidia-smi -L || true; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,memory.used --format=csv; md5sum analysis/bench_proteomelm.py $IN/proteins.fasta ext/ProteomeLM/data/interactomes/enhanced_ppi_model_dscript.pt; wc -l $IN/pairs.csv
test -e $OUT/bench_proteomelm_h200.json && { echo "output exists"; exit 0; }
rm -rf $OUT; mkdir -p $OUT; cd ext/ProteomeLM
../pm_env/bin/python -u ../../analysis/bench_proteomelm.py $IN $OUT --pairs 14000000 --chunk 1000000
rc=$?; echo "exit $rc"; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv; exit $rc
