#!/usr/bin/env python3
"""Score one species' test pairs with ProteomeLM-PPI as released: ProteomeLM-S backbone + the author-bundled supervised head (enhanced_ppi_model_dscript.pt, trained by the authors on D-SCRIPT human).
Uses the repo's own notebook functions unchanged. Run from the repository root (ext/ProteomeLM) with ext/pm_env/bin/python.
usage: score_pm.py FASTA PAIRS_CSV OUT_TSV [--limit N] [--head data/interactomes/enhanced_ppi_model_dscript.pt] [--backbone Bitbol-Lab/ProteomeLM-S]"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from proteomelm.ppi.notebook_inference import (load_proteomelm_backbone, prepare_notebook_proteome, load_supervised_ppi_model, iter_pair_chunks, score_pair_chunks)

ap = argparse.ArgumentParser(); ap.add_argument("fasta"); ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--head", default="data/interactomes/enhanced_ppi_model_dscript.pt"); ap.add_argument("--backbone", default="Bitbol-Lab/ProteomeLM-S"); ap.add_argument("--cache", default=None); a = ap.parse_args()
t0 = time.time(); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
P = pd.read_csv(a.pairs, dtype=str); P["label"] = P["label"].astype(int)
if a.limit: P = P.iloc[:a.limit].copy()
cache = a.cache or (os.path.splitext(a.out)[0] + "_esmc.pt")
backbone = load_proteomelm_backbone(a.backbone, device=dev)
proteome = prepare_notebook_proteome(a.backbone, a.fasta, encoded_genome_file=cache, esm_device="cuda:0" if dev == "cuda" else "cpu", proteomelm_device=dev, model=backbone)
labels = list(proteome.labels); idx = {l: i for i, l in enumerate(labels)}; print(f"proteome: {len(labels)} proteins ({(time.time() - t0) / 60:.1f} min)", flush=True)
miss = (set(P["query"]) | set(P["text"])) - set(idx); assert not miss, ("pair proteins missing from the FASTA", list(miss)[:5])
sup = load_supervised_ppi_model(a.head, protein_embed_dim=int(proteome.protein_embeddings.shape[1]), pair_feature_dim=1, device=dev, expected_backbone=a.backbone)
ia = P["query"].map(idx).to_numpy(); ib = P["text"].map(idx).to_numpy(); explicit = [(int(x), int(y)) for x, y in zip(ia, ib) if x != y]
chunks = iter_pair_chunks(len(labels), chunk_size=max(len(explicit), 1), explicit_pairs=explicit)
R = score_pair_chunks(backbone, proteome, chunks, device=dev, logistic_model=None, supervised_model=sup)
key = {(int(r.idx_a), int(r.idx_b)): (float(r.unsupervised_score), float(r.supervised_score)) for r in R.itertuples()}
lo, hi = np.minimum(ia, ib), np.maximum(ia, ib); sc = [key.get((int(x), int(y)), (np.nan, np.nan)) for x, y in zip(lo, hi)]
P["unsupervised_score"] = [s[0] for s in sc]; P["supervised_score"] = [s[1] for s in sc]; assert P.supervised_score.notna().all() and np.isfinite(P.supervised_score).all()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); P.to_csv(a.out, sep="\t", index=False)
from sklearn.metrics import roc_auc_score, average_precision_score
print(f"wrote {a.out}: {len(P)} pairs; AUROC supervised {roc_auc_score(P.label, P.supervised_score):.3f} AUPR {average_precision_score(P.label, P.supervised_score):.3f}; AUROC unsupervised {roc_auc_score(P.label, P.unsupervised_score):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
