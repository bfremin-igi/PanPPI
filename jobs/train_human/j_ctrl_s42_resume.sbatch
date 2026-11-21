#!/bin/bash
#SBATCH --job-name=ctrl42r
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
# Same run dir, same flags/seed/batch (batch.cfg of the original run), plus --resume; code = main_ctrl_t120.py (the original code with ONLY the NCCL collective timeout raised 10 -> 120 min, one line, numerics identical; the original crashed once on a 25 min stall).
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
cd <SCRATCH>
PY=./ppi_env/bin/python
DATA=data/logoppi/data
RUN=runs_ft/ctrl_lg_650M_800_s42
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/$RUN/nccl_trace_${SLURM_JOB_ID}_
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 0.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 42 --diag-species yeast,ecoli,mouse,224308"
CFG=$RUN/batch.cfg
date; hostname; nvidia-smi -L || true
[ -f "$RUN/output/checkpoints/last.pt" ] || { echo "no last.pt in $RUN: nothing to resume; refusing"; exit 1; }
[ -f "$CFG" ] || { echo "no batch.cfg in $RUN; refusing"; exit 1; }
ls -la --time-style=full-iso "$RUN/output/checkpoints/last.pt"
cat "$RUN/output/results.txt" | cut -c1-120
md5sum main_ctrl_t120.py threebranch_e3.py
echo "===== smoke 0a: shared E3 gradient-isolation unit tests"
$PY tests/test_e3_head.py || { echo "SMOKE 0a FAILED"; exit 1; }
echo "===== smoke 0b: E3a --no-sn --no-gp unit tests"
$PY tests/test_e3a_nogp.py || { echo "SMOKE 0b FAILED"; exit 1; }
source $CFG
echo "===== CTRL seed 42 RESUME: batch $BATCH x accum $ACCUM x 2 GPUs"
nvidia-smi dmon -s pucvmet -d 30 -o T > "$RUN/gpu_dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl_t120.py $E3A --run-dir "$RUN" \
  --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200 --resume
rc=$?
echo "exit $rc"
exit $rc
