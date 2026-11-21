#!/bin/bash
#SBATCH --job-name=ctrl42b
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
# CTRL, SEED 42: a second seed of "LoGoPPI trained in OUR harness". IDENTICAL to jobs/j_ctrl.sbatch (job 41253852, seed 141) except: --seed 42, its own run dir, no --requeue, exit code propagated,
# NCCL flight recorder to scratch, nvidia-smi dmon, refuses to start if the run dir already holds a last.pt (a clean seed run starts at epoch 1).
# WHY: (1) the harness's own seed noise, to read CTRL (seed 141) vs released LoGoPPI (yeast -.019 was borderline vs a ~.01 seed SD); (2) the CTRL half of the paired E3a-vs-CTRL comparison at seed 42.
PY=./ppi_env/bin/python
export HF_HOME=<SCRATCH>/.hf
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
cd <SCRATCH>
DATA=data/logoppi/data
RUN=runs_ft/ctrl_lg_650M_800_s42_r2
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/$RUN/nccl_trace_${SLURM_JOB_ID}_
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 0.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 42 --diag-species yeast,ecoli,mouse,224308"
CFG=$RUN/batch.cfg
date; hostname; nvidia-smi -L || true
mkdir -p "$RUN/output/checkpoints"
[ -f "$RUN/output/checkpoints/last.pt" ] && { echo "last.pt already exists in $RUN: a clean run must start at epoch 1; refusing"; exit 1; }
md5sum main_ctrl_t120.py threebranch_e3.py

echo "===== smoke 0a: shared E3 gradient-isolation unit tests"
$PY tests/test_e3_head.py || { echo "SMOKE 0a FAILED"; exit 1; }
echo "===== smoke 0b: CTRL --no-sn --no-gp unit tests"
$PY tests/test_e3a_nogp.py || { echo "SMOKE 0b FAILED"; exit 1; }
BATCH=24; ACCUM=2
echo "===== smoke 1: memory probe batch 24 x 800"
if ! $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl_t120.py $E3A --run-dir runs_ft/ctrl42b_smoke_$SLURM_JOB_ID \
     --mem-probe --batch $BATCH; then
  BATCH=16; ACCUM=3
  echo "===== batch 24 does not fit; probing batch $BATCH"
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl_t120.py $E3A --run-dir runs_ft/ctrl42b_smoke_$SLURM_JOB_ID \
    --mem-probe --batch $BATCH || { echo "SMOKE 1 FAILED at batch 16 too"; exit 1; }
fi
echo "BATCH=$BATCH ACCUM=$ACCUM" > $CFG
echo "===== smoke 2: tiny 2-rank cycle INCLUDING the post-training report"
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl_t120.py $E3A --run-dir runs_ft/ctrl42b_smoke_$SLURM_JOB_ID \
  --epochs 1 --batch 8 --accum 1 --subset 512 --eval-subset 300 --progress-every 20 --final-species yeast \
  || { echo "SMOKE 2 FAILED"; exit 1; }
rm -rf runs_ft/ctrl42b_smoke_$SLURM_JOB_ID
echo "===== smoke passed"

source $CFG
echo "===== CTRL seed 42: 14-epoch ceiling, batch $BATCH x accum $ACCUM x 2 GPUs = $((BATCH*ACCUM*2))"
nvidia-smi dmon -s pucvmet -d 30 -o T > "$RUN/gpu_dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl_t120.py $E3A --run-dir "$RUN" \
  --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200
rc=$?
echo "exit $rc"
exit $rc
