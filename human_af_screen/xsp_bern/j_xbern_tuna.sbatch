#!/bin/bash
#SBATCH --job-name=xbtu
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SETS=(yeast mouse fly worm ecoli); SP=${SETS[$SLURM_ARRAY_TASK_ID]}; PY=./ppi_env/bin/python; SD=ext/TUnA/results/bernett/TUnA; W=$SD/output/model; D=data/xsp_bern/data/test; B=data/xsp_bern/conv; OUT=analysis_out/xsp_bern/tuna
date; hostname; nvidia-smi -L || true; md5sum analysis/prep_xsp_bern.py analysis/xsp_tuna.py analysis/xsp_convert.py $W
test -s $OUT/$SP.csv && { echo "output exists"; exit 0; }
mkdir -p $OUT; $PY -I analysis/prep_xsp_bern.py || { echo "PREP FAILED"; echo "exit 1"; exit 1; }
$PY -I analysis/xsp_convert.py --data-dir $D --out-dir $B --species $SP || { echo "CONVERT FAILED"; echo "exit 1"; exit 1; }
n=$(wc -l < $B/$SP/test_interaction.tsv); echo "converted pairs: $n"
echo "===== smoke: 64 pairs (NOT results)"
$PY -u analysis/xsp_tuna.py --seed-dir $SD --weights $W --data-root $B --species $SP --esm-dir ext/esm2_150M_hf --out-dir $OUT/_smoke_$SP --vendored-dir analysis --smoke || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf $OUT/_smoke_$SP
$PY -u analysis/xsp_tuna.py --seed-dir $SD --weights $W --data-root $B --species $SP --esm-dir ext/esm2_150M_hf --out-dir $OUT/_work_$SP --vendored-dir analysis || { echo "SCORING FAILED"; echo "exit 1"; exit 1; }
m=$(( $(wc -l < $OUT/_work_$SP/$SP.csv) - 1 )); echo "rows scored: $m"; [ "$m" -eq "$n" ] || { echo "ROW COUNT MISMATCH ($m vs $n)"; echo "exit 4"; exit 4; }
mv $OUT/_work_$SP/$SP.csv $OUT/$SP.csv; echo "exit 0"; exit 0
