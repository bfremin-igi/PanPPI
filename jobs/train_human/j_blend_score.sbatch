#!/bin/bash
#SBATCH --job-name=blendsc
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# Scores with each run's own final-report procedure: E3a clean2 epoch_05 + epoch_06 (analysis/score_ckpt_e3.py, imports main_e3 146ddae7) and CTRL s42 epoch_06 (analysis/score_ckpt_ctrl.py, main_ctrl aee4c0ff).
# GATES (a run's numbers are not interpreted if its gate fails): E3a epoch_05 human_selection must reproduce the training log (logoppi_g+m .9517, global .9604) within +-0.002 AUPR;
# CTRL s42 epoch_06 must reproduce its log (g+m .9434, global .9519) within +-0.002.  A smoke pass runs first in this allocation.
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python
E3=runs_ft/e3a_nounc_650M_800_s141_clean2/output/checkpoints
C42=runs_ft/ctrl_lg_650M_800_s42/output/checkpoints
date; hostname; nvidia-smi -L || true
md5sum main_e3.py main_ctrl.py threebranch_e3.py analysis/score_ckpt_e3.py analysis/score_ckpt_ctrl.py
ls -la $E3/epoch_05.pt $E3/epoch_06.pt $C42/epoch_06.pt || { echo "missing checkpoint"; exit 1; }
echo "===== smoke E3a (tiny subsets, yeast only; NOT results)"
$PY -I analysis/score_ckpt_e3.py --ckpt $E3/epoch_05.pt --out-dir analysis_out/e3a_ckpt_score --smoke || { echo "SMOKE FAILED"; exit 1; }
echo "===== E3a epoch_05 (gate) and epoch_06"
$PY -I analysis/score_ckpt_e3.py --ckpt $E3/epoch_05.pt,$E3/epoch_06.pt --out-dir analysis_out/e3a_ckpt_score || { echo "E3a SCORING FAILED"; exit 1; }
echo "===== CTRL s42 epoch_06"
$PY -I analysis/score_ckpt_ctrl.py --ckpt $C42/epoch_06.pt --out-dir analysis_out/ctrl42_ckpt_score --seed 42 || { echo "CTRL42 SCORING FAILED"; exit 1; }
echo "===== gates"
$PY -I - <<'PYEOF'
import pandas as pd, sys
ok = True
def get(f, m, s="human_selection"):
    d = pd.read_csv(f); return float(d[(d.set == s) & (d.model == m)].aupr.iloc[0])
for f, g, gl, nm in (("analysis_out/e3a_ckpt_score/epoch_05.csv", .9517, .9604, "E3a ep5"), ("analysis_out/ctrl42_ckpt_score/epoch_06.csv", .9434, .9519, "CTRL s42 ep6")):
    a, b = get(f, "logoppi_g+m"), get(f, "global")
    p = abs(a - g) <= .002 and abs(b - gl) <= .002
    print(f"GATE {nm}: g+m {a:.4f} (log {g}) global {b:.4f} (log {gl}) -> {'PASS' if p else 'FAIL'}"); ok &= p
sys.exit(0 if ok else 3)
PYEOF
rc=$?
echo "exit $rc"
exit $rc
