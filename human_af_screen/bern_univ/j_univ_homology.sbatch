#!/bin/bash
#SBATCH --job-name=univhom
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# Output analysis_out/univ_homology/{hits.m8, maxid.tsv} (maxid_cov50 = best identity among hits with query coverage >= 0.5, as homhum2). Propagates the exit code.
set -uo pipefail
cd <SCRATCH>
MM=<SCRATCH_HOME>/localcolabfold/colabfold-conda/bin/mmseqs; PY=./ppi_env/bin/python; O=analysis_out/univ_homology; TG=analysis_out/homology/human_train_proteins.faa
test -e $O/maxid.tsv && { echo "$O/maxid.tsv exists, refusing"; echo "exit 1"; exit 1; }
mkdir -p $O logs; rm -rf $O/tmp
date; hostname; $MM version; md5sum $TG
$PY -I - $O <<'PYEOF' || { echo "INPUT BUILD FAILED"; echo "exit 1"; exit 1; }
import sys
O = sys.argv[1]; q = {}
for sp in ("ecoli", "mtb", "paer", "yeast", "fly", "mouse", "worm"):
    k = None
    for l in open(f"data/univ/{sp}/proteins.fasta"):
        l = l.strip()
        if l.startswith(">"): k = l[1:].split()[0]; assert k not in q, ("duplicate id across species", k); q[k] = ""
        elif l: q[k] += l
open(f"{O}/queries.faa", "w").write("".join(f">{k}\n{v.rstrip('*')}\n" for k, v in sorted(q.items()))); print("query proteins:", len(q))
PYEOF
md5sum $O/queries.faa
mkdir -p $O/tmp
$MM easy-search $O/queries.faa $TG $O/hits.m8.tmp $O/tmp -s 7.5 -e 1e-3 --max-seqs 500 --threads 16 --format-output "query,target,pident,qcov,tcov,evalue,bits" || { echo "MMSEQS FAILED"; echo "exit 1"; exit 1; }
mv $O/hits.m8.tmp $O/hits.m8; wc -l $O/hits.m8
$PY -I - $O <<'PYEOF' || { echo "SUMMARY FAILED"; echo "exit 1"; exit 1; }
import sys, pandas as pd
O = sys.argv[1]
q = [l[1:].split()[0] for l in open(f"{O}/queries.faa") if l.startswith(">")]
h = pd.read_csv(f"{O}/hits.m8", sep="\t", header=None, names=["query", "target", "pident", "qcov", "tcov", "evalue", "bits"], dtype={"query": str, "target": str})
if h["pident"].max() > 1.5: h["pident"] = h["pident"] / 100.0
a = h.groupby("query")["pident"].max().rename("maxid_any"); b = h[h["qcov"] >= 0.5].groupby("query")["pident"].max().rename("maxid_cov50"); n = h.groupby("query").size().rename("n_hits")
t = pd.DataFrame({"query": q}).merge(a, on="query", how="left").merge(b, on="query", how="left").merge(n, on="query", how="left")
t["n_hits"] = t["n_hits"].fillna(0).astype(int); t.to_csv(f"{O}/maxid.tsv", sep="\t", index=False)
print("proteins", len(t), "| with any hit", int(t.maxid_any.notna().sum()), "| with cov>=0.5 hit", int(t.maxid_cov50.notna().sum()), "| with cov>=0.5 identity>=0.4", int((t.maxid_cov50 >= 0.4).sum()))
PYEOF
rm -rf $O/tmp
echo "exit 0"; exit 0
