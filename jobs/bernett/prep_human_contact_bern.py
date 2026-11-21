#!/usr/bin/env python3
"""Pair list for the Bernett-trained models on the human contact-verified candidate set (PLAN.md 'BERNETT-TRAINED PanPPI AND LoGoPPI ON THE HUMAN CONTACT-VERIFIED PAIRS').
Reads analysis_out/bern_cand2/human/cand_scores.tsv.gz (tier pdb, variant A, level L0, all replicates), maps i/j to accessions via the screen FASTA order (md5 of the id list is asserted),
writes data/humanaf_bern/{human_pairs.csv (query,text,label), human_pairs_ij.tsv (i,j,a,b,label), human.fasta (symlink), bchk files (symlinks)}. Idempotent; new files only. Run from the FastPPI2 root."""
import os, hashlib, pandas as pd
ids = [l[1:].split()[0] for l in open("data/humanaf/screen_proteins.fasta") if l.startswith(">")]
assert len(ids) == 19408 and hashlib.md5(("\n".join(ids) + "\n").encode()).hexdigest() == "1f15a5769c6e7fae93b41fe5ce807c8e", "id list differs from the one the i/j indices refer to"
d = pd.read_csv("analysis_out/bern_cand2/human/cand_scores.tsv.gz", sep="\t", usecols=["i", "j", "label", "tier", "level", "variant"])
d = d[(d.tier == "pdb") & (d.variant == "A") & (d.level == "L0")]; assert d.groupby(["i", "j"]).label.nunique().max() == 1
d = d.drop_duplicates(["i", "j"]).reset_index(drop=True); d["a"] = d.i.map(lambda x: ids[x]); d["b"] = d.j.map(lambda x: ids[x]); print("unique pairs:", len(d), "positives:", int(d.label.sum()))
O = "data/humanaf_bern"; os.makedirs(O, exist_ok=True)
d[["i", "j", "a", "b", "label"]].to_csv(f"{O}/human_pairs_ij.tsv", sep="\t", index=False)
d.rename(columns={"a": "query", "b": "text"})[["query", "text", "label"]].to_csv(f"{O}/human_pairs.csv", index=False)
for src, dst in (("data/humanaf/screen_proteins.fasta", "human.fasta"), ("data/xsp_bern/data/test/bchk.fasta", "bchk.fasta"), ("data/xsp_bern/data/test/bchk_pairs.csv", "bchk_pairs.csv")):
    p = f"{O}/{dst}"
    if not os.path.lexists(p): os.symlink(os.path.abspath(src), p)
print("ok")
