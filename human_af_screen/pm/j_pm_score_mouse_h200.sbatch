#!/bin/bash
#SBATCH --job-name=pmmouse
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>
export HF_HOME=<SCRATCH>/.hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
SPS=(yeast mouse fly worm ecoli 208964 224308 83332); SP=${SPS[$SLURM_ARRAY_TASK_ID]}
D=<SCRATCH>/data/logoppi/data/test; OUT=<SCRATCH>/analysis_out/xsp_pm; mkdir -p $OUT
test -s $OUT/$SP.tsv && { echo "$SP: output exists"; exit 0; }
date; hostname; nvidia-smi -L || true; echo "species $SP: proteins $(grep -c '>' $D/$SP.fasta), pairs $(($(wc -l < $D/${SP}_pairs.csv) - 1))"; md5sum ext/ProteomeLM/data/interactomes/enhanced_ppi_model_dscript.pt $D/$SP.fasta $D/${SP}_pairs.csv analysis/score_pm.py
cd ext/ProteomeLM
echo "===== smoke (NOT results)"; rm -rf $OUT/_smoke_$SP; mkdir -p $OUT/_smoke_$SP
../pm_env/bin/python -u ../../analysis/score_pm.py $D/$SP.fasta $D/${SP}_pairs.csv $OUT/_smoke_$SP/s.tsv --limit 1000 || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf $OUT/_smoke_$SP
echo "===== full $SP"
../pm_env/bin/python -u ../../analysis/score_pm.py $D/$SP.fasta $D/${SP}_pairs.csv $OUT/$SP.tsv
rc=$?; echo "exit $rc"; exit $rc
