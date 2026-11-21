#!/bin/bash
#SBATCH --job-name=lgbranch
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
# test files of yeast, mouse, fly, worm, ecoli (data/logoppi/data/test); task 5 = LoGoPPI-Bernett on the Bernett test file (52,048 pairs). Inference only. Propagates the exit code.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
NAMES=(yeast mouse fly worm ecoli bernett); NM=${NAMES[$SLURM_ARRAY_TASK_ID]}; OUT=analysis_out/lg_branch
if [ $NM = bernett ]; then M=ext/LoGoPPI-Bernett; FA=ext/LoGoPPI-Bernett/data/bernett/proteins.fasta; PC=ext/LoGoPPI-Bernett/data/bernett/test.csv
else M=ext/LoGoPPI-Cross-species; FA=data/logoppi/data/test/$NM.fasta; PC=data/logoppi/data/test/${NM}_pairs.csv; fi
date; hostname; nvidia-smi -L || true; md5sum $M/model.safetensors $FA $PC ext/LoGoPPI/inference.py
test -s $OUT/$NM.csv && { echo "output exists"; exit 0; }
mkdir -p $OUT; n=$(( $(wc -l < $PC) - 1 )); echo "pairs: $n"
./lg_env/bin/python ext/LoGoPPI/inference.py --model_dir $M --fasta_path $FA --pair_csv $PC --output_path $OUT/_tmp_$NM.csv --gpus 0 --detailed
rc=$?; [ $rc -eq 0 ] || { echo "exit $rc"; exit $rc; }
m=$(( $(wc -l < $OUT/_tmp_$NM.csv) - 1 )); echo "rows scored: $m"; [ "$m" -eq "$n" ] || { echo "ROW COUNT MISMATCH ($m vs $n)"; echo "exit 4"; exit 4; }
mv $OUT/_tmp_$NM.csv $OUT/$NM.csv; echo "exit 0"; exit 0
