#!/bin/bash
#SBATCH --job-name=xbpan2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --time=04:00:00
#SBATCH --exclude=n0134.savio4
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
# the stored Bernett-test scores of the same run (enforced, exit 3 if > 5e-3) so the model construction is verified. Propagates the exit code.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SEEDS=(42 7 141); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; RUN=runs_ft/e3a_bernett_650M_800_s${S}_h1; OUT=analysis_out/xsp_bern/panppi_s$S
date; hostname; nvidia-smi -L || true; md5sum analysis/xsp_components_bern2.py analysis/prep_xsp_bern.py data/xsp_bern/pairs_all.tsv.gz $RUN/output/checkpoints/best.pt $RUN/calibration.json
test -s $OUT/components_ecoli.tsv && { echo "output exists"; exit 0; }
./ppi_env/bin/python -I analysis/prep_xsp_bern.py || { echo "PREP FAILED"; echo "exit 1"; exit 1; }
./ppi_env/bin/python -I analysis/xsp_components_bern2.py $RUN data/xsp_bern/data/test $OUT --tags bchk,yeast,mouse,fly,worm,ecoli --check $RUN/scores_bernett.tsv
rc=$?; echo "exit $rc"; exit $rc
