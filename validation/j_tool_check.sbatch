#!/bin/bash
#SBATCH --job-name=toolchk
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
set -uo pipefail
export HF_HOME=<SCRATCH>/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd <SCRATCH>
PY=./ppi_env/bin/python; RUN=runs_ft/e3a_nounc_650M_800_s42_r2; ESM=ext/esm2_650M_hf; OUT=analysis_out/tool_check; export PYTHONPATH=$PWD/panppi_tool
CK=$RUN/output/checkpoints/best.pt; CAL=$RUN/calibration.json; V=panppi_tool/validation
rc=0; fail() { echo "STEP FAILED: $1"; rc=1; }
date; hostname; nvidia-smi -L || true; mkdir -p $OUT; md5sum panppi_tool/panppi/*.py panppi_tool/examples/* $V/tool_check_compare.py
echo "===== md5 of the six checkpoints (for the weights table)"; for r in e3a_nounc_650M_800_s42_r2:human_s42 e3a_nounc_650M_800_s7:human_s7 e3a_nounc_650M_800_s141_clean2:human_s141 e3a_bernett_650M_800_s42_h1:bernett_s42 e3a_bernett_650M_800_s7_h1:bernett_s7 e3a_bernett_650M_800_s141_h1:bernett_s141; do d=runs_ft/${r%%:*}; n=${r##*:}; echo "weights_${n}_best.pt $(md5sum < $d/output/checkpoints/best.pt | cut -d' ' -f1) $(stat -c %s $d/output/checkpoints/best.pt)"; echo "weights_${n}_calibration.json $(md5sum < $d/calibration.json | cut -d' ' -f1)"; done
echo "===== ESM folder"; grep -i "_name_or_path\|architectures\|\"esm\|vocab_size\|num_hidden_layers\|hidden_size\|position_embedding" $ESM/config.json | head -8; ls $ESM
echo "===== 0. unit tests (CPU, stand-in encoder)"; $PY panppi_tool/tests/test_panppi.py 2>&1 | grep "PASS\|FAIL\|SKIP\|ALL\|Error" | tail -40 || fail unit
echo "===== 1. examples (5 pairs)"; $PY -u -m panppi --fasta panppi_tool/examples/proteins.fasta --pairs panppi_tool/examples/pairs.csv --weights $CK --calibration $CAL --esm $ESM --out $OUT/examples_scores.csv || fail examples
cat $OUT/examples_scores.csv
echo "===== 2. 3,000 yeast test pairs vs the stored scores of the same run (fp32 tool vs stored final report)"
$PY - <<'PYEOF'
import pandas as pd
d = pd.read_csv("data/logoppi/data/test/yeast_pairs.csv").sample(3000, random_state=0).rename(columns={"query": "a", "text": "b"})
d[["a", "b"]].to_csv("analysis_out/tool_check/yeast3000_pairs.csv", index=False)
PYEOF
$PY -u -m panppi --fasta data/logoppi/data/test/yeast.fasta --pairs $OUT/yeast3000_pairs.csv --weights $CK --calibration $CAL --esm $ESM --out $OUT/yeast3000_scores.csv || fail yeast
$PY - <<'PYEOF'
import pandas as pd
d = pd.read_csv("analysis_out/tool_check/yeast3000_pairs.csv"); l = pd.read_csv("data/logoppi/data/test/yeast_pairs.csv").rename(columns={"query": "a", "text": "b"})
d.merge(l, on=["a", "b"]).to_csv("analysis_out/tool_check/yeast3000_pairs_labelled.csv", index=False)
PYEOF
$PY $V/tool_check_compare.py $OUT/yeast3000_scores.csv $RUN/scores_yeast.tsv || fail compare
echo "===== 3. all-vs-all on 600 random human proteins, --top 1000, vs pairs mode on all 179,700 pairs"
$PY - <<'PYEOF'
import numpy as np, itertools, pandas as pd
ids, seqs, k = [], {}, None
for l in open("data/humanaf/screen_proteins.fasta"):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; seqs[k] = ""
    elif k: seqs[k] += l
pick = sorted(np.random.RandomState(0).choice(sorted(seqs), 600, replace=False))
open("analysis_out/tool_check/h600.fasta", "w").write("".join(f">{p}\n{seqs[p]}\n" for p in pick))
pd.DataFrame(list(itertools.combinations(pick, 2)), columns=["a", "b"]).to_csv("analysis_out/tool_check/h600_allpairs.csv", index=False)
PYEOF
$PY -u -m panppi --fasta $OUT/h600.fasta --all-vs-all --top 1000 --weights $CK --calibration $CAL --esm $ESM --out $OUT/h600_top1000.csv 2>&1 | tail -8 || fail allvsall
$PY -u -m panppi --fasta $OUT/h600.fasta --pairs $OUT/h600_allpairs.csv --weights $CK --calibration $CAL --esm $ESM --out $OUT/h600_pairs_mode.csv 2>&1 | tail -3 || fail pairsmode
$PY - <<'PYEOF'
import pandas as pd, numpy as np
a = pd.read_csv("analysis_out/tool_check/h600_top1000.csv"); p = pd.read_csv("analysis_out/tool_check/h600_pairs_mode.csv").sort_values("blend_logit", ascending=False).head(1000)
ka = set(zip(a.protein_a, a.protein_b)); kp = set(zip(p.protein_a, p.protein_b)); print(f"top-1000 sets: {len(ka & kp)} of 1000 shared (ties at the cut may differ)")
m = a.merge(p, on=["protein_a", "protein_b"], suffixes=("_a", "_p")); print(f"max |blend diff| on shared pairs {(m.blend_logit_a - m.blend_logit_p).abs().max():.2e}")
print("ALLVSALL OK" if len(ka & kp) >= 995 else "ALLVSALL MISMATCH")
PYEOF
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
date; echo "exit $rc"; exit $rc
