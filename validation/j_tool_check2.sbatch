#!/bin/bash
#SBATCH --job-name=toolchk2
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python; ESM=ext/esm2_650M_hf; OUT=analysis_out/tool_check; export PYTHONPATH=$PWD/panppi_tool; V=panppi_tool/validation
H=runs_ft/e3a_nounc_650M_800_s42_r2; BR=runs_ft/e3a_bernett_650M_800_s42_h1; rc=0; fail() { echo "STEP FAILED: $1"; rc=1; }
date; hostname; nvidia-smi -L || true; md5sum panppi_tool/panppi/score.py panppi_tool/panppi/weights.py $V/tool_check_compare.py
sample() { $PY - "$@" <<'PYEOF'
import sys, pandas as pd
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); d = pd.read_csv(src).rename(columns={"query": "a", "text": "b"}); d.sample(n, random_state=0)[["a", "b"]].to_csv(out, index=False)
PYEOF
}
echo "===== A. yeast 3,000 pairs, --bf16 (the precision of the stored run) vs stored scores"
sample data/logoppi/data/test/yeast_pairs.csv $OUT/yeast3000_pairs_b.csv 3000
$PY -u -m panppi --fasta data/logoppi/data/test/yeast.fasta --pairs $OUT/yeast3000_pairs_b.csv --weights $H/output/checkpoints/best.pt --calibration $H/calibration.json --esm $ESM --bf16 --out $OUT/yeast3000_bf16.csv 2>&1 | tail -2 || fail A
$PY $V/tool_check_compare.py $OUT/yeast3000_bf16.csv $H/scores_yeast.tsv || fail A_compare
echo "===== B. E. coli 3,000 pairs, --bf16, vs stored"
sample data/logoppi/data/test/ecoli_pairs.csv $OUT/ecoli3000_pairs.csv 3000
$PY -u -m panppi --fasta data/logoppi/data/test/ecoli.fasta --pairs $OUT/ecoli3000_pairs.csv --weights $H/output/checkpoints/best.pt --calibration $H/calibration.json --esm $ESM --bf16 --out $OUT/ecoli3000_bf16.csv 2>&1 | tail -2 || fail B
$PY $V/tool_check_compare.py $OUT/ecoli3000_bf16.csv $H/scores_ecoli.tsv || fail B_compare
echo "===== C. Bernett model (seed 42): 3,000 Bernett test pairs, --bf16, vs stored"
ls data/bernett/data/test | head -4
sample data/bernett/data/test/bernett_pairs.csv $OUT/bernett3000_pairs.csv 3000
$PY -u -m panppi --fasta data/bernett/data/test/bernett.fasta --pairs $OUT/bernett3000_pairs.csv --weights $BR/output/checkpoints/best.pt --calibration $BR/calibration.json --esm $ESM --bf16 --out $OUT/bernett3000_bf16.csv 2>&1 | tail -2 || fail C
$PY $V/tool_check_compare.py $OUT/bernett3000_bf16.csv $BR/scores_bernett.tsv || fail C_compare
echo "===== D. throughput and peak GPU memory: all-vs-all on 3,000 random human proteins (4.5M pairs), --top 1000"
$PY - <<'PYEOF'
import numpy as np
seqs, k = {}, None
for l in open("data/humanaf/screen_proteins.fasta"):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; seqs[k] = ""
    elif k: seqs[k] += l
pick = sorted(np.random.RandomState(1).choice(sorted(seqs), 3000, replace=False)); open("analysis_out/tool_check/h3000.fasta", "w").write("".join(f">{p}\n{seqs[p]}\n" for p in pick))
PYEOF
$PY -u - <<PYEOF 2>&1 | grep -v "^  embedded\|^  anchor [0-9]*[1-9]/" | tail -12
import time, torch
from panppi.score import main
t = time.time(); main(["--fasta", "$OUT/h3000.fasta", "--all-vs-all", "--top", "1000", "--weights", "$H/output/checkpoints/best.pt", "--calibration", "$H/calibration.json", "--esm", "$ESM", "--out", "$OUT/h3000_top1000.csv"])
print(f"TOTAL wall {time.time()-t:.0f} s for 3,000 proteins / 4,498,500 pairs (includes model load, embedding and the self-check); peak GPU memory allocated {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PYEOF
echo "===== E. batch-size independence: 600-protein all-vs-all with --batch 64 vs the first job's output (--batch 256)"
$PY -u -m panppi --fasta $OUT/h600.fasta --all-vs-all --top 1000 --batch 64 --weights $H/output/checkpoints/best.pt --calibration $H/calibration.json --esm $ESM --out $OUT/h600_top1000_b64.csv 2>&1 | tail -2 || fail E
$PY - <<'PYEOF'
import pandas as pd
a = pd.read_csv("analysis_out/tool_check/h600_top1000.csv"); b = pd.read_csv("analysis_out/tool_check/h600_top1000_b64.csv"); m = a.merge(b, on=["protein_a", "protein_b"], suffixes=("_1", "_2"))
print(f"shared pairs {len(m)} of 1000; max |blend diff| {(m.blend_logit_1 - m.blend_logit_2).abs().max():.2e}"); print("BATCH OK" if len(m) >= 995 and (m.blend_logit_1 - m.blend_logit_2).abs().max() < 1e-3 else "BATCH MISMATCH")
PYEOF
echo "===== F. CPU path: the 5 example pairs with no GPU (fp32) vs the first job's GPU output"
CUDA_VISIBLE_DEVICES="" $PY -u -m panppi --fasta panppi_tool/examples/proteins.fasta --pairs panppi_tool/examples/pairs.csv --weights $H/output/checkpoints/best.pt --calibration $H/calibration.json --esm $ESM --device cpu --out $OUT/examples_cpu.csv 2>&1 | tail -3 || fail F
$PY - <<'PYEOF'
import pandas as pd
c = pd.read_csv("analysis_out/tool_check/examples_cpu.csv"); g = pd.read_csv("analysis_out/tool_check/examples_scores.csv"); m = c.merge(g, on=["protein_a", "protein_b"], suffixes=("_c", "_g"))
d = (m.blend_logit_c - m.blend_logit_g).abs().max(); print(f"CPU vs GPU max |blend diff| {d:.2e}"); print("CPU OK" if d < 5e-3 and len(m) == 5 else "CPU MISMATCH")
PYEOF
echo "===== G. all six models on the five example pairs (BID-MCL1, MCM6-CDT1, TP53-MDM2 interact; BID-MCM6, MCL1-CDT1 do not)"
for r in e3a_nounc_650M_800_s42_r2:human_s42 e3a_nounc_650M_800_s7:human_s7 e3a_nounc_650M_800_s141_clean2:human_s141 e3a_bernett_650M_800_s42_h1:bernett_s42 e3a_bernett_650M_800_s7_h1:bernett_s7 e3a_bernett_650M_800_s141_h1:bernett_s141; do d=runs_ft/${r%%:*}; n=${r##*:}
  $PY -u -m panppi --fasta panppi_tool/examples/proteins.fasta --pairs panppi_tool/examples/pairs.csv --weights $d/output/checkpoints/best.pt --calibration $d/calibration.json --esm $ESM --out $OUT/examples_$n.csv > /dev/null 2>&1 || fail G_$n
done
$PY - <<'PYEOF'
import pandas as pd
names = ["human_s42", "human_s7", "human_s141", "bernett_s42", "bernett_s7", "bernett_s141"]; ok = True; tab = {}
for n in names:
    d = pd.read_csv(f"analysis_out/tool_check/examples_{n}.csv"); tab[n] = d.probability.round(3).tolist(); pos, neg = d.blend_logit[[0, 1, 2]].min(), d.blend_logit[[3, 4]].max(); ok &= bool(pos > neg)
print(pd.DataFrame(tab, index=["BID-MCL1", "MCM6-CDT1", "TP53-MDM2", "BID-MCM6", "MCL1-CDT1"]).to_string()); print("ORDERING OK" if ok else "ORDERING CHECK FAILED (interacting pairs not all above non-partners for every model)")
PYEOF
date; echo "exit $rc"; exit $rc
