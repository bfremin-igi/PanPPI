#!/bin/bash
#SBATCH --job-name=blendsc42
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# Scores E3a seed-42 (run dir ..._s42_r2) epoch_05 ONLY with the run's own final-report procedure (analysis/score_ckpt_e3.py, imports main_e3 146ddae7), --seed 42.
# GATE (numbers not interpreted if it fails): human_selection logoppi_g+m .9441 and global .9584 (training log, epoch 5) within +-0.002 AUPR.  A smoke pass runs first in this allocation.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python
E3=runs_ft/e3a_nounc_650M_800_s42_r2/output/checkpoints
date; hostname; nvidia-smi -L || true
md5sum main_e3.py threebranch_e3.py analysis/score_ckpt_e3.py
ls -la $E3/epoch_05.pt || { echo "missing checkpoint"; exit 1; }
echo "===== smoke E3a (tiny subsets, yeast only; NOT results)"
$PY -I analysis/score_ckpt_e3.py --ckpt $E3/epoch_05.pt --out-dir analysis_out/e3a42_ckpt_score_smoke --smoke --seed 42 || { echo "SMOKE FAILED"; exit 1; }
echo "===== E3a seed 42 epoch_05"
$PY -I analysis/score_ckpt_e3.py --ckpt $E3/epoch_05.pt --out-dir analysis_out/e3a42_ckpt_score --seed 42 || { echo "SCORING FAILED"; exit 1; }
echo "===== gate"
$PY -I - <<'PYEOF'
import pandas as pd, sys
d = pd.read_csv("analysis_out/e3a42_ckpt_score/epoch_05.csv")
g = lambda m: float(d[(d.set == "human_selection") & (d.model == m)].aupr.iloc[0])
a, b = g("logoppi_g+m"), g("global")
p = abs(a - .9441) <= .002 and abs(b - .9584) <= .002
print(f"GATE E3a s42 ep5: g+m {a:.4f} (log .9441) global {b:.4f} (log .9584) -> {'PASS' if p else 'FAIL'}")
sys.exit(0 if p else 3)
PYEOF
rc=$?
echo "exit $rc"
exit $rc
