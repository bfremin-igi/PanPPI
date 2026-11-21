#!/bin/bash
#SBATCH --job-name=e3ar
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# E3a RESUME (new file; jobs/j_e3a.sbatch is untouched). 41254062 died in epoch 3 (355,200/421,312 pairs) of a NCCL ALLREDUCE watchdog timeout on node n4259.savio4:
# rank 0 had queued collectives up to #1286286 while rank 1 had queued only up to #1286185 and never issued #1286186, i.e. rank 1 STALLED before communicating.
# That is not a graph desync (the rank loss has no collective; the zero-positive branch returns xa.sum()*0 which keeps every parameter in the graph) -- cause of the stall is UNKNOWN
# (GPU/node fault is the leading suspect). This script changes NOTHING about training: same code (main_e3.py md5 146ddae7, threebranch_e3.py 89d4e0ba), same flags, same RUN dir,
# resumes from last.pt (written at the end of epoch 2, 2026-11-12 22:09). DIFFERENCES, all observational:
#  1. exit status is the training command's (the old wrapper ended in `echo "exit $?"`, so Slurm said COMPLETED 0:0 for a crash);
#  2. NCCL flight recorder ON, dumped to scratch (the default path was /global/home/..., over quota), so a repeat shows which collective rank 1 last issued;
#  3. nvidia-smi dmon every 30 s (power/temp/util/clocks/violations/memory/ECC+PCIe errors/PCIe throughput) into the run dir, to see whether a GPU degrades before a stall;
#  4. refuses to start from scratch: no last.pt -> exit 1 (starting over is a different decision).
# A resumed run is NOT bit-identical to an uninterrupted one: the per-epoch length-grouping RNG is reseeded from (seed, start_epoch, rank). Same distribution, different batch order from epoch 3 on.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
cd <SCRATCH>
PY=./ppi_env/bin/python
DATA=data/logoppi/data
RUN=runs_ft/e3a_nounc_650M_800_s141
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$PWD/$RUN/nccl_trace_${SLURM_JOB_ID}_
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 1.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 141 --diag-species yeast,ecoli,mouse,224308"
CFG=$RUN/batch.cfg
date; hostname; nvidia-smi -L || true
[ -f "$RUN/output/checkpoints/last.pt" ] || { echo "NO last.pt: refusing to start E3a from scratch"; exit 1; }
[ -f "$CFG" ] || { echo "NO batch.cfg"; exit 1; }
md5sum main_e3.py threebranch_e3.py
source $CFG
echo "===== E3a RESUME: 14 epochs, batch $BATCH x accum $ACCUM x 2 GPUs = $((BATCH*ACCUM*2))"
nvidia-smi dmon -s pucvmet -d 30 -o T > "$RUN/gpu_dmon_${SLURM_JOB_ID}.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_e3.py $E3A --run-dir "$RUN" --resume \
  --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200
rc=$?
echo "exit $rc"
exit $rc
