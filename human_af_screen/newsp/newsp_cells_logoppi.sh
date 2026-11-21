#!/bin/bash
# Released LoGoPPI-Cross-species on the unique cell pairs of a species. usage: newsp_cells_logoppi.sh SPECIES. cwd = FastPPI2.
set -uo pipefail
SP=$1; LG=./lg_env/bin/python; PY=./ppi_env/bin/python; M=ext/LoGoPPI-Cross-species; D=data/newsp/$SP; O=analysis_out/newsp/$SP
$PY - "$O" <<'PYEOF'
import sys, pandas as pd
O = sys.argv[1]; p = pd.read_csv(O + "/cell_pairs.csv", dtype=str); pd.DataFrame({"query": p.protein_a, "text": p.protein_b, "label": 0}).to_csv(O + "/cell_pairs_lg.csv", index=False)
PYEOF
$LG ext/LoGoPPI/inference.py --model_dir $M --fasta_path $D/cand_proteins.fasta --pair_csv $O/cell_pairs_lg.csv --output_path $O/cells_logoppi.csv --gpus 0
rc=$?; echo "exit $rc"; exit $rc
