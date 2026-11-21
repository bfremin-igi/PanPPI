#!/bin/bash
#SBATCH --job-name=e3a42b
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# E3a, SEED 42, clean uninterrupted, from epoch 1, in its own run dir. The E3a half of the paired comparison with CTRL seed 42 (41288219). Submitted with --dependency=afterok:41288101:41288219
# (ORIGINAL dependency text, superseded: this resubmission has NO dependency.)
# TRAINING IDENTICAL to jobs/j_e3a_clean.sbatch (41288101): same code (main_e3_t120.py md5 146ddae7, threebranch_e3.py 89d4e0ba), same flags, 14-epoch ceiling, patience 3, blend selection; ONLY --seed 42 and the run dir differ.
# Seed 42 is from the pre-registered {42, 7} list and is LoGoPPI's own seed; chosen before any result. No --requeue (a preemption ends the run, never silently resumes it); refuses to start if last.pt exists;
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
cd <SCRATCH>
PY=./ppi_env/bin/python
DATA=data/logoppi/data
RUN=runs_ft/e3a_nounc_650M_800_s42_r2
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/$RUN/nccl_trace_${SLURM_JOB_ID}_
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 1.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 42 --diag-species yeast,ecoli,mouse,224308"
CFG=$RUN/batch.cfg
date; hostname; nvidia-smi -L || true
mkdir -p "$RUN/output/checkpoints"
[ -f "$RUN/output/checkpoints/last.pt" ] && { echo "last.pt already exists in $RUN: a CLEAN run must start at epoch 1; refusing"; exit 1; }
md5sum main_e3_t120.py threebranch_e3.py

echo "===== smoke 0a: shared E3 gradient-isolation unit tests"
$PY tests/test_e3_head.py || { echo "SMOKE 0a FAILED"; exit 1; }
echo "===== smoke 0b: E3a --no-sn --no-gp unit tests"
$PY tests/test_e3a_nogp.py || { echo "SMOKE 0b FAILED"; exit 1; }
BATCH=24; ACCUM=2
echo "===== smoke 1: memory probe batch 24 x 800"
if ! $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_e3_t120.py $E3A --run-dir runs_ft/e3a42b_smoke_$SLURM_JOB_ID \
     --mem-probe --batch $BATCH; then
  BATCH=16; ACCUM=3
  echo "===== batch 24 does not fit; probing batch $BATCH"
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_e3_t120.py $E3A --run-dir runs_ft/e3a42b_smoke_$SLURM_JOB_ID \
    --mem-probe --batch $BATCH || { echo "SMOKE 1 FAILED at batch 16 too"; exit 1; }
fi
echo "BATCH=$BATCH ACCUM=$ACCUM" > $CFG
echo "===== smoke 2: tiny 2-rank cycle INCLUDING the post-training report"
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_e3_t120.py $E3A --run-dir runs_ft/e3a42b_smoke_$SLURM_JOB_ID \
  --epochs 1 --batch 8 --accum 1 --subset 512 --eval-subset 300 --progress-every 20 --final-species yeast \
  || { echo "SMOKE 2 FAILED"; exit 1; }
rm -rf runs_ft/e3a42b_smoke_$SLURM_JOB_ID
echo "===== smoke passed"

source $CFG
echo "===== E3a CLEAN: 14-epoch ceiling, batch $BATCH x accum $ACCUM x 2 GPUs = $((BATCH*ACCUM*2))"
nvidia-smi dmon -s pucvmet -d 30 -o T > "$RUN/gpu_dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_e3_t120.py $E3A --run-dir "$RUN" \
  --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200
rc=$?
echo "exit $rc"
exit $rc
