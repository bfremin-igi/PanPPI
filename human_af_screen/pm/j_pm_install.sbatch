#!/bin/bash
#SBATCH --job-name=pminstall
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
cd <SCRATCH>
export HF_HOME=<SCRATCH>/.hf PIP_CACHE_DIR=<SCRATCH>/.pipcache TMPDIR=<SCRATCH>/.tmp; mkdir -p $TMPDIR $PIP_CACHE_DIR
PYB=<SCRATCH_HOME>/localcolabfold/conda/bin/python3; R=ext/ProteomeLM
date; hostname; $PYB --version; test -d $R || { echo "repo missing"; echo "exit 1"; exit 1; }
test -d ext/pm_env && { echo "ext/pm_env exists, refusing"; echo "exit 1"; exit 1; }
$PYB -m venv --system-site-packages ext/pm_env || { echo "VENV FAILED"; echo "exit 1"; exit 1; }
ext/pm_env/bin/pip install --upgrade pip 2>&1 | tail -n 1
ext/pm_env/bin/pip install -r $R/requirements.txt 2>&1 | tail -n 15 || { echo "PIP REQUIREMENTS FAILED"; echo "exit 1"; exit 1; }
ext/pm_env/bin/pip install -e $R --no-deps 2>&1 | tail -n 3
ext/pm_env/bin/pip freeze > ext/pm_env_freeze.txt; grep -i "^torch==\|^transformers==\|^esm==\|^peft==\|^numpy==\|^scikit-learn==" ext/pm_env_freeze.txt
echo "===== downloads + import check"
cd $R && ../pm_env/bin/python - <<'PYEOF' || { echo "IMPORT/DOWNLOAD FAILED"; echo "exit 1"; exit 1; }
import torch
from esm.models.esmc import ESMC
m = ESMC.from_pretrained("esmc_600m"); print("ESM-C 600M loaded, params %.0fM" % (sum(p.numel() for p in m.parameters()) / 1e6)); del m
from proteomelm import ProteomeLMForMaskedLM
p = ProteomeLMForMaskedLM.from_pretrained("Bitbol-Lab/ProteomeLM-S"); print("ProteomeLM-S loaded, params %.1fM" % (sum(x.numel() for x in p.parameters()) / 1e6))
import proteomelm.ppi.notebook_inference as ni; print("notebook_inference import ok; torch", torch.__version__)
PYEOF
echo "exit 0"
