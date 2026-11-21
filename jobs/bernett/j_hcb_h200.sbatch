#!/bin/bash
#SBATCH --job-name=hcbern
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A.out
# Stage 1 prep; stage 2 PanPPI Bernett seeds 42/7/141 (tags bchk = the first 2,000 Bernett test pairs, checked against the run's stored scores with the rank criterion, then human); stage 3 LoGoPPI-Bernett --detailed.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>; D=data/humanaf_bern; fail=0
date; hostname; nvidia-smi -L || true; md5sum analysis/prep_human_contact_bern.py analysis/xsp_components_bern2.py ext/LoGoPPI/inference.py ext/LoGoPPI-Bernett/model.safetensors
./ppi_env/bin/python -I analysis/prep_human_contact_bern.py || { echo "PREP FAILED"; echo "exit 1"; exit 1; }
for S in 42 7 141; do
  RUN=runs_ft/e3a_bernett_650M_800_s${S}_h1; OUT=analysis_out/humanaf_bern/panppi_s$S
  test -s $OUT/components_human.tsv && { echo "seed $S exists"; continue; }
  ./ppi_env/bin/python -I analysis/xsp_components_bern2.py $RUN $D $OUT --tags bchk,human --check $RUN/scores_bernett.tsv; rc=$?; echo "panppi seed $S exit $rc"; [ $rc -eq 0 ] || fail=1
done
OUT=analysis_out/humanaf_bern/logoppi; M=ext/LoGoPPI-Bernett; mkdir -p $OUT
if [ ! -s $OUT/human.csv ]; then
  n=$(( $(wc -l < $D/human_pairs.csv) - 1 )); echo "pairs: $n"
  ./lg_env/bin/python ext/LoGoPPI/inference.py --model_dir $M --fasta_path $D/human.fasta --pair_csv $D/human_pairs.csv --output_path $OUT/_tmp_human.csv --gpus 0 --detailed; rc=$?
  if [ $rc -eq 0 ]; then m=$(( $(wc -l < $OUT/_tmp_human.csv) - 1 )); echo "rows scored: $m"; if [ "$m" -eq "$n" ]; then mv $OUT/_tmp_human.csv $OUT/human.csv; else echo "ROW COUNT MISMATCH"; fail=1; fi; else echo "logoppi exit $rc"; fail=1; fi
fi
echo "exit $fail"; exit $fail
