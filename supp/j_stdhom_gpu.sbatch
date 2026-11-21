#!/bin/bash
#SBATCH --job-name=stdhom
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=<SCRATCH>/logs/%x_%j.out
# to any protein of the HUMAN TRAINING set (analysis_out/homology/human_train_proteins.faa, the target of homhum2/univhom), EXISTING mmseqs2 binary, same settings as those (-s 7.5 -e 1e-3, more hits kept). CPU only on a GPU-node slot
# (the GPU is idle; the CPU partition was full for the AFM search). Writes only analysis_out/stdhom/. Propagates the exit code.
set -uo pipefail
cd <SCRATCH>
MM=<SCRATCH_HOME>/localcolabfold/colabfold-conda/bin/mmseqs; O=analysis_out/stdhom; TG=analysis_out/homology/human_train_proteins.faa; T=data/logoppi/data/test
test -e $O/hits.m8 && { echo "$O/hits.m8 exists, refusing"; echo "exit 1"; exit 1; }
mkdir -p $O logs; rm -rf $O/tmp; mkdir -p $O/tmp
date; hostname; $MM version; md5sum $TG
for sp in yeast mouse fly worm ecoli; do test -s $T/$sp.fasta || { echo "missing $T/$sp.fasta"; echo "exit 1"; exit 1; }; done
cat $T/yeast.fasta $T/mouse.fasta $T/fly.fasta $T/worm.fasta $T/ecoli.fasta | sed 's/\*$//' > $O/queries.faa
grep -c ">" $O/queries.faa $TG; grep ">" $O/queries.faa | sort | uniq -d | head -3
$MM easy-search $O/queries.faa $TG $O/hits.m8.tmp $O/tmp -s 7.5 -e 1e-3 --max-seqs 500 --threads 4 --format-output "query,target,pident,qcov,tcov,evalue,bits" || { echo "MMSEQS FAILED"; echo "exit 1"; exit 1; }
mv $O/hits.m8.tmp $O/hits.m8; wc -l $O/hits.m8; rm -rf $O/tmp
echo "exit 0"; exit 0
