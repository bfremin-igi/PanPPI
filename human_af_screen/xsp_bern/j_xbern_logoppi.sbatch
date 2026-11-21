#!/bin/bash
#SBATCH --job-name=xblg
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SETS=(yeast mouse fly worm ecoli); SP=${SETS[$SLURM_ARRAY_TASK_ID]}; D=data/xsp_bern/data/test; OUT=analysis_out/xsp_bern/logoppi; M=ext/LoGoPPI-Bernett
date; hostname; nvidia-smi -L || true; md5sum analysis/prep_xsp_bern.py $M/model.safetensors ext/LoGoPPI/inference.py
test -s $OUT/$SP.csv && { echo "output exists"; exit 0; }
mkdir -p $OUT; ./ppi_env/bin/python -I analysis/prep_xsp_bern.py || { echo "PREP FAILED"; echo "exit 1"; exit 1; }
n=$(( $(wc -l < $D/${SP}_pairs.csv) - 1 )); echo "pairs: $n"
./lg_env/bin/python ext/LoGoPPI/inference.py --model_dir $M --fasta_path $D/$SP.fasta --pair_csv $D/${SP}_pairs.csv --output_path $OUT/_tmp_$SP.csv --gpus 0 --detailed
rc=$?; [ $rc -eq 0 ] || { echo "exit $rc"; exit $rc; }
m=$(( $(wc -l < $OUT/_tmp_$SP.csv) - 1 )); echo "rows scored: $m"; [ "$m" -eq "$n" ] || { echo "ROW COUNT MISMATCH"; echo "exit 4"; exit 4; }
mv $OUT/_tmp_$SP.csv $OUT/$SP.csv; echo "exit 0"; exit 0
