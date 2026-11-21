#!/bin/bash
#SBATCH --job-name=xbpi
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
SETS=(yeast mouse fly worm ecoli); SP=${SETS[$SLURM_ARRAY_TASK_ID]}; PY=./ppi_env/bin/python; D=data/xsp_bern/data/test; OUT=analysis_out/xsp_bern/plminteract; X=ext_bernett_pi; I=$OUT/_in_$SP; W=$OUT/_work_$SP
date; hostname; nvidia-smi -L || true; md5sum analysis/prep_xsp_bern.py analysis/zeroshot_external.py ext/weights/plminteract_bernett/pytorch_model.bin; ls -la $X/weights
test -s $OUT/$SP.csv && { echo "output exists"; exit 0; }
mkdir -p $OUT $I; $PY -I analysis/prep_xsp_bern.py || { echo "PREP FAILED"; echo "exit 1"; exit 1; }
$PY -I - $D/${SP}_pairs.csv $D/$SP.fasta $I <<'PYEOF' || { echo "INPUT BUILD FAILED"; echo "exit 1"; exit 1; }
import sys, pandas as pd
t, f, o = sys.argv[1:4]
d, k = {}, None
for l in open(f):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; d[k] = ""
    elif l: d[k] += l
p = pd.read_csv(t, dtype=str); p["label"] = p["label"].astype(int)
miss = (set(p["query"]) | set(p["text"])) - set(d); assert not miss, ("proteins missing", list(miss)[:3])
p[["query", "text", "label"]].to_csv(f"{o}/test_interaction.tsv", sep="\t", header=False, index=False)
pd.DataFrame({"rec_id": p["query"], "pep_id": p["text"], "label": p["label"]}).to_csv(f"{o}/test_meta.tsv", sep="\t", index=False)
used = sorted(set(p["query"]) | set(p["text"]))
open(f"{o}/dict.tsv", "w").write("".join(f"{k}\t{d[k]}\n" for k in used)); print("pairs", len(p), "proteins", len(used))
PYEOF
echo "===== smoke: 32 pairs (NOT results)"
S=$OUT/_smoke_$SP; rm -rf $S; mkdir -p $S
$PY -u analysis/zeroshot_external.py --model plminteract --ext-dir $X --data-dir $I --dictionary $I/dict.tsv --out-dir $S --parts test --limit 32 || { echo "SMOKE FAILED"; echo "exit 1"; exit 1; }
rm -rf $S; mkdir -p $W
$PY -u analysis/zeroshot_external.py --model plminteract --ext-dir $X --data-dir $I --dictionary $I/dict.tsv --out-dir $W --parts test || { echo "SCORING FAILED (work dir kept for resume)"; echo "exit 1"; exit 1; }
$PY -I - $W $I/test_interaction.tsv $OUT/$SP.csv <<'PYEOF' || { echo "COLLECT FAILED"; echo "exit 1"; exit 1; }
import sys, os, pandas as pd
W, inter, out = sys.argv[1:4]
r = pd.read_csv(f"{W}/evaluation_results_test.tsv", sep="\t"); p = pd.read_csv(inter, sep="\t", header=None, names=["query", "text", "label"], dtype=str)
assert len(r) == len(p), ("row count", len(r), len(p)); assert (r["Protein A"].astype(str).values == p["query"].values).all() and (r["Protein B"].astype(str).values == p["text"].values).all(), "row order differs"; assert r["S"].notna().all(), "NaN scores"
pd.DataFrame({"query": p["query"], "text": p["text"], "score": r["S"]}).to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out); print("collected", len(r), "rows ->", out)
PYEOF
echo "exit 0"; exit 0
