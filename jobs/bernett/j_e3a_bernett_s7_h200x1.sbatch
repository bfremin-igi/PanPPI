#!/bin/bash
#SBATCH --job-name=e3abern7h1
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# (train Intra1 163,192 pairs; validation Intra0 halved into selection / calibration; test Intra2 52,048 pairs, 1:1). ONLY changes: --pos-weight 1.0 (the Bernett data is 1:1; LoGoPPI-Bernett also used pos weight 1; our human runs used 10 for 1:10),
# data dir data/bernett/data, --diag-species / --final-species bernett, code = main_e3_bernett.py (copy of main_e3_t120.py with the pos weight made a flag, default 10.0 = unchanged). Smokes in the same allocation. Exit code propagated.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
cd <SCRATCH>
PY=./ppi_env/bin/python; DATA=data/bernett/data; RUN=runs_ft/e3a_bernett_650M_800_s7_h1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/$RUN/nccl_trace_${SLURM_JOB_ID}_
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 1.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 7 --pos-weight 1.0 --diag-species bernett --final-species bernett"
CFG=$RUN/batch.cfg
date; hostname; nvidia-smi -L || true
mkdir -p "$RUN/output/checkpoints"
[ -f "$RUN/output/checkpoints/last.pt" ] && echo "last.pt exists: RESUMING (a resubmission after an interruption)"
md5sum main_e3_bernett.py threebranch_e3.py $DATA/human_train_clean.csv $DATA/test/bernett_pairs.csv
echo "===== smoke 0a/0b: unit tests"
$PY tests/test_e3_head.py || { echo "SMOKE 0a FAILED"; exit 1; }
$PY tests/test_e3a_nogp.py || { echo "SMOKE 0b FAILED"; exit 1; }
BATCH=24; ACCUM=4
echo "===== smoke 1: memory probe batch 24 x 800"
if ! $PY -m torch.distributed.run --standalone --nproc_per_node=1 main_e3_bernett.py $E3A --run-dir runs_ft/e3abern_smoke_$SLURM_JOB_ID --mem-probe --batch $BATCH; then
  BATCH=16; ACCUM=6
  echo "===== batch 24 does not fit; probing batch $BATCH"
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 main_e3_bernett.py $E3A --run-dir runs_ft/e3abern_smoke_$SLURM_JOB_ID --mem-probe --batch $BATCH || { echo "SMOKE 1 FAILED at batch 16 too"; exit 1; }
fi
[ -f $CFG ] && source $CFG || echo "BATCH=$BATCH ACCUM=$ACCUM" > $CFG
echo "===== smoke 2: tiny 2-rank cycle INCLUDING the post-training report"
$PY -m torch.distributed.run --standalone --nproc_per_node=1 main_e3_bernett.py $E3A --run-dir runs_ft/e3abern_smoke_$SLURM_JOB_ID \
  --epochs 1 --batch 8 --accum 1 --subset 512 --eval-subset 300 --progress-every 20 || { echo "SMOKE 2 FAILED"; exit 1; }
rm -rf runs_ft/e3abern_smoke_$SLURM_JOB_ID
echo "===== smoke passed"
source $CFG
echo "===== E3a on Bernett: 14-epoch ceiling, batch $BATCH x accum $ACCUM x 1 GPU"
nvidia-smi dmon -s pucvmet -d 30 -o T > "$RUN/gpu_dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT
$PY -m torch.distributed.run --standalone --nproc_per_node=1 main_e3_bernett.py $E3A --run-dir "$RUN" --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200 --resume
rc=$?
echo "exit $rc"
exit $rc
