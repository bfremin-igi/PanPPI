#!/bin/bash
# MMseqs2 search of the candidate proteins of one species against the human TRAINING proteins (same settings and target as univ_homology / homhum2). usage: newsp_homology.sh SPECIES. cwd = FastPPI2.
set -uo pipefail
SP=$1; MM=<SCRATCH_HOME>/localcolabfold/colabfold-conda/bin/mmseqs; PY=./ppi_env/bin/python; TG=analysis_out/homology/human_train_proteins.faa; Q=data/newsp/$SP/cand_proteins.fasta; O=analysis_out/newsp/$SP/homology
test -e $O/maxid.tsv && { echo "$O/maxid.tsv exists, refusing"; exit 1; }
mkdir -p $O/tmp; $MM version; md5sum $TG $Q
$MM easy-search $Q $TG $O/hits.m8.tmp $O/tmp -s 7.5 -e 1e-3 --max-seqs 500 --threads 8 --format-output "query,target,pident,qcov,tcov,evalue,bits" || { echo "MMSEQS FAILED"; exit 1; }
mv $O/hits.m8.tmp $O/hits.m8; wc -l $O/hits.m8
$PY -I - $O $Q <<'PYEOF'
import sys, pandas as pd
O, Q = sys.argv[1:3]; q = [l[1:].split()[0] for l in open(Q) if l.startswith(">")]
h = pd.read_csv(f"{O}/hits.m8", sep="\t", header=None, names=["query", "target", "pident", "qcov", "tcov", "evalue", "bits"], dtype={"query": str, "target": str})
if len(h) and h["pident"].max() > 1.5: h["pident"] = h["pident"] / 100.0
a = h.groupby("query")["pident"].max().rename("maxid_any"); b = h[h["qcov"] >= 0.5].groupby("query")["pident"].max().rename("maxid_cov50"); n = h.groupby("query").size().rename("n_hits")
t = pd.DataFrame({"query": q}).merge(a, on="query", how="left").merge(b, on="query", how="left").merge(n, on="query", how="left"); t.to_csv(f"{O}/maxid.tsv", sep="\t", index=False)
print(f"{len(t):,} proteins; with a hit {int(t.maxid_any.notna().sum()):,}; maxid_cov50 >= 0.4: {int((t.maxid_cov50 >= 0.4).sum()):,}")
PYEOF
rm -rf $O/tmp; rc=$?; echo "exit $rc"; exit $rc
