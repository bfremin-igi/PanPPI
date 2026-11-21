#!/bin/bash
# Released LoGoPPI-Cross-species for the multi-species screen. usage: newsp_logoppi.sh SPECIES SUBTASK   (SUBTASK 0-2: block pairs of the 2-block all-pairs scoring of the subset; 3: the AFM-confident pairs). cwd = FastPPI2.
set -uo pipefail
SP=$1; ST=$2; LG=./lg_env/bin/python; PY=./ppi_env/bin/python; M=ext/LoGoPPI-Cross-species; D=data/newsp/$SP; O=analysis_out/newsp/$SP; mkdir -p $O
$PY - "$SP" <<'PYEOF'
import sys, os
sp = sys.argv[1]; D = f"data/newsp/{sp}"; O = f"analysis_out/newsp/{sp}"
seqs, k = {}, None
for l in open(D + "/needed.fasta"):
    l = l.strip()
    if l.startswith(">"): k = l[1:].split()[0]; seqs[k] = ""
    elif l: seqs[k] += l
sub = [l.strip() for l in open(D + "/subset_ids.txt") if l.strip()]
if not os.path.exists(O + "/subset.fasta"): open(O + "/subset.fasta", "w").write("".join(f">{a}\n{seqs[a]}\n" for a in sub))
import pandas as pd
a = pd.read_csv(D + "/afm_pass_pairs.csv"); pd.DataFrame({"query": a.a, "text": a.b, "label": 1}).to_csv(O + "/afm_pairs.csv", index=False)
PYEOF
if [ "$ST" -lt 3 ]; then
  $PY -u analysis/run_logoppi_universe.py $O/subset.fasta $O/logoppi_subset $LG $M $ST --blocks 2
else
  $LG ext/LoGoPPI/inference.py --model_dir $M --fasta_path $D/needed.fasta --pair_csv $O/afm_pairs.csv --output_path $O/logoppi_afm.csv --gpus 0
fi
rc=$?; echo "exit $rc"; exit $rc
