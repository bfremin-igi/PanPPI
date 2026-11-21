#!/bin/bash
#SBATCH --job-name=univclu
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --time=01:00:00
#SBATCH --output=<SCRATCH>/logs/%x_%A_%a.out
set -uo pipefail
cd <SCRATCH>
MM=<SCRATCH_HOME>/localcolabfold/colabfold-conda/bin/mmseqs
SPS=(human ecoli mtb paer yeast fly mouse worm); SP=${SPS[$SLURM_ARRAY_TASK_ID]}; O=analysis_out/bern_univ_v2/$SP
if [ $SP = human ]; then FA=data/humanaf/screen_proteins.fasta; else FA=data/univ/$SP/proteins.fasta; fi
date; hostname; $MM version; md5sum $FA
test -s $O/clusters_id40_cluster.tsv && { echo "output exists"; exit 0; }
rm -rf $O/_clutmp; mkdir -p $O/_clutmp
$MM easy-cluster $FA $O/clusters_id40 $O/_clutmp --min-seq-id 0.4 -c 0.5 --threads 8
rc=$?; [ $rc -eq 0 ] || { echo "exit $rc"; exit $rc; }
rm -rf $O/_clutmp; wc -l $O/clusters_id40_cluster.tsv; cut -f1 $O/clusters_id40_cluster.tsv | sort -u | wc -l
echo "exit 0"; exit 0
