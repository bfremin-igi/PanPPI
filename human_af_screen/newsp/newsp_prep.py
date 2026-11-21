"""Prepare the multi-species screen inputs. For each species: universe = UniProt reference proteome + candidate proteins fetched by accession; AFDB candidates (STRING physical links) inside the universe; AFM-confident pairs;
candidate degree (for the hub control); a fixed random subset of M proteins (seed 11) whose all-pairs score distribution estimates the rank fractions of the AFM-confident pairs; needed.fasta = subset + proteins of AFM-confident pairs.
Writes newsp/data/<sp>/{needed.fasta, subset_ids.txt, afm_pass_pairs.csv, cand_degree.tsv, meta.json}."""
import json, os, numpy as np, pandas as pd
SP = {"soybean": 3847, "arabidopsis": 3702, "rat": 10116, "rice": 39947, "zebrafish": 7955}; M = 6000
T = pd.read_csv("../afdb/hetero_slim.tsv.gz", sep="\t", usecols=["uniprot_ac_1", "uniprot_ac_2", "tax_id_1", "tax_id_2", "ipTM", "max_ipSAE", "passes"])
def read(path):
    d, k = {}, None
    if not os.path.exists(path): return d
    for l in open(path):
        l = l.strip()
        if l.startswith(">"): h = l[1:].split()[0]; k = h.split("|")[1] if "|" in h else h; d[k] = ""
        elif l and k: d[k] += l.rstrip("*")
    return d
for n, tx in SP.items():
    seqs = read(f"proteomes/{n}.fasta"); seqs.update(read(f"proteomes/{n}_extra.fasta")); d = T[(T.tax_id_1 == tx) & (T.tax_id_2 == tx)]
    d = d[d.uniprot_ac_1 != d.uniprot_ac_2].drop_duplicates(["uniprot_ac_1", "uniprot_ac_2"]); deg = pd.concat([d.uniprot_ac_1, d.uniprot_ac_2]).value_counts()   # degree from the WHOLE candidate graph, sequence or not
    ok = d.uniprot_ac_1.isin(seqs) & d.uniprot_ac_2.isin(seqs); lost = int((~ok).sum()); P = d[ok & d.passes]; n_pass_all = int(d.passes.sum())
    ids = sorted(seqs); rs = np.random.RandomState(11); sub = sorted(rs.choice(ids, min(M, len(ids)), replace=False).tolist())
    need = sorted(set(sub) | set(P.uniprot_ac_1) | set(P.uniprot_ac_2)); o = f"data/{n}"; os.makedirs(o, exist_ok=True)
    with open(f"{o}/needed.fasta", "w") as f:
        for a in need: f.write(f">{a}\n{seqs[a]}\n")
    open(f"{o}/subset_ids.txt", "w").write("\n".join(sub) + "\n")
    P[["uniprot_ac_1", "uniprot_ac_2", "ipTM", "max_ipSAE"]].rename(columns={"uniprot_ac_1": "a", "uniprot_ac_2": "b"}).to_csv(f"{o}/afm_pass_pairs.csv", index=False)
    deg.rename_axis("protein").rename("degree").reset_index().to_csv(f"{o}/cand_degree.tsv", sep="\t", index=False)
    meta = dict(species=n, taxon=tx, universe_proteins=len(ids), universe_pairs=len(ids) * (len(ids) - 1) // 2, candidates=len(d), candidates_without_sequence=lost, afm_confident_all=n_pass_all, afm_confident=len(P), subset=len(sub), needed_proteins=len(need))
    json.dump(meta, open(f"{o}/meta.json", "w")); print(meta, flush=True)
