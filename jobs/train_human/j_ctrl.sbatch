#!/bin/bash
#SBATCH --job-name=ctrl
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
# CTRL = "LoGoPPI trained in OUR harness": the matched baseline for the E3 vs LoGoPPI comparison.
# Identical to j_e3a.sbatch except --rank-loss-weight 0.0 (NO rank loss), main_ctrl.py, and its own RUN dir.
# So: LoGoPPI's two branches (Global BCE + untrained MaxSim), no latent, no SN, no GP; length-grouped batching,
# --select-on blend, seed 141, --epochs 14, same data/splits/recipe as E3 (41249819).
# main_ctrl.py is main_e3.py with ONE change (freeze proj_m when the rank-loss weight is 0): proj_m would
# otherwise never reach the loss and DDP(find_unused_parameters=False) errors. main_e3.py is NOT touched
# (E3 and E3a re-read it at runtime). Smoke 2 takes several optimizer steps under 2-rank DDP, so a failure of the
# freeze shows up there. Smoke 0a/0b re-run the head unit tests (they test the head, not the weight-0 path).
# Expected: blend lands near run D's g+m (yeast .7372 / ecoli .7622) and released LoGoPPI (.7464 / .7595).
# If not, the harness moves the result and every E1/E3 gain is suspect. Reports NO variance (nan is expected).
PY=./ppi_env/bin/python
DATA="${DATA:?set DATA to the LoGoPPI bundle data/ dir}"
RUN=runs_ft/ctrl_lg_650M_800_s141
E3A="--esm-dir ext/esm2_650M_hf --data-dir $DATA --no-latent --no-grad-ckpt --length-grouped --rank-loss-weight 0.0 --rank-temp 1.0 --select-on blend --no-sn --no-gp --seed 141 --diag-species yeast,ecoli,mouse,224308"
CFG=$RUN/batch.cfg
date; nvidia-smi -L || true
mkdir -p "$RUN/output/checkpoints"

if [ ! -f "$RUN/output/checkpoints/last.pt" ]; then
  # Smoke 0, both suites. test_e3_head.py asserts the shared architecture (gradient isolation on the GP
  # configuration); test_e3a_nogp.py re-asserts all three isolation claims with sn/gp OFF -- the config this
  # job actually trains -- and proves the post-hoc uncertainty path is fittable and faithful to the logit.
  echo "===== smoke 0a: shared E3 gradient-isolation unit tests (18 checks)"
  $PY tests/test_e3_head.py || { echo "SMOKE 0a FAILED"; exit 1; }
  echo "===== smoke 0b: CTRL --no-sn --no-gp unit tests (22 checks)"
  $PY tests/test_e3a_nogp.py || { echo "SMOKE 0b FAILED"; exit 1; }
  BATCH=24; ACCUM=2
  echo "===== smoke 1: memory probe batch 24 x 800"
  if ! $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl.py $E3A --run-dir runs_ft/ctrl_smoke_$SLURM_JOB_ID \
       --mem-probe --batch $BATCH; then
    BATCH=16; ACCUM=3
    echo "===== batch 24 does not fit; probing batch $BATCH"
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl.py $E3A --run-dir runs_ft/ctrl_smoke_$SLURM_JOB_ID \
      --mem-probe --batch $BATCH || { echo "SMOKE 1 FAILED at batch 16 too"; exit 1; }
  fi
  echo "BATCH=$BATCH ACCUM=$ACCUM" > $CFG
  # Smoke 2 exercises the whole post-training report on the no-GP path, which is where job 41247085 died.
  echo "===== smoke 2: tiny 2-rank cycle INCLUDING the post-training report"
  $PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl.py $E3A --run-dir runs_ft/ctrl_smoke_$SLURM_JOB_ID \
    --epochs 1 --batch 8 --accum 1 --subset 512 --eval-subset 300 --progress-every 20 --final-species yeast \
    || { echo "SMOKE 2 FAILED"; exit 1; }
  rm -rf runs_ft/ctrl_smoke_$SLURM_JOB_ID
  echo "===== smoke passed"
fi

source $CFG
echo "===== CTRL: 14 epochs, batch $BATCH x accum $ACCUM x 2 GPUs = $((BATCH*ACCUM*2))"
$PY -m torch.distributed.run --standalone --nproc_per_node=2 main_ctrl.py $E3A --run-dir "$RUN" --resume \
  --epochs 14 --batch "$BATCH" --accum "$ACCUM" --progress-every 200
echo "exit $?"
