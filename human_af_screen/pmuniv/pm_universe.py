#!/usr/bin/env python3
"""ProteomeLM-PPI (released ProteomeLM-S + bundled D-SCRIPT-trained head) on ALL pairs i<j of a protein set, in pair-index order (pair k <-> (i,j) with k = i*N - i(i+1)/2 + (j-i-1)).
Repo functions unchanged: extract_attention_pair_features -> score_pairs_supervised. Chunked, resume-safe. Run from the repository root (ext/ProteomeLM) with ext/pm_env python.
usage: pm_universe.py FASTA IDS_TXT OUT_DIR [--chunk 4000000]"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from proteomelm.ppi.notebook_inference import (load_proteomelm_backbone, prepare_notebook_proteome, load_supervised_ppi_model, extract_attention_pair_features, score_pairs_supervised)
ap = argparse.ArgumentParser(); ap.add_argument("fasta"); ap.add_argument("ids"); ap.add_argument("out"); ap.add_argument("--chunk", type=int, default=4_000_000)
ap.add_argument("--backbone", default="Bitbol-Lab/ProteomeLM-S"); ap.add_argument("--head", default="data/interactomes/enhanced_ppi_model_dscript.pt"); a = ap.parse_args()
dev = "cuda"; sync = lambda: torch.cuda.synchronize(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); log = lambda m: print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)
ids = [l.strip() for l in open(a.ids) if l.strip()]; N = len(ids); NT = N * (N - 1) // 2; log(f"{N} proteins, {NT:,} pairs")
STARTS = np.arange(N, dtype=np.int64) * N - np.arange(N, dtype=np.int64) * (np.arange(N, dtype=np.int64) + 1) // 2
def untri(k): i = np.searchsorted(STARTS, k, side="right") - 1; return i, i + 1 + (k - STARTS[i])
backbone = load_proteomelm_backbone(a.backbone, device=dev); sync(); t = time.time()
proteome = prepare_notebook_proteome(a.backbone, a.fasta, encoded_genome_file=os.path.join(a.out, "esmc_cache.pt"), esm_device="cuda:0", proteomelm_device=dev, model=backbone, reload_if_possible=True); sync()
assert list(proteome.labels) == ids, "FASTA order differs from ids.txt"; embed_s = time.time() - t; log(f"proteome ready ({embed_s:.0f}s for ESM-C embedding + forward)")
sup = load_supervised_ppi_model(a.head, protein_embed_dim=int(proteome.protein_embeddings.shape[1]), pair_feature_dim=1, device=dev, expected_backbone=a.backbone)
sp = os.path.join(a.out, "scores.f64"); pp = os.path.join(a.out, "progress.txt"); S = np.memmap(sp, dtype=np.float64, mode=("r+" if os.path.exists(sp) else "w+"), shape=(NT,)); start = int(open(pp).read()) if os.path.exists(pp) else 0
nchunk = (NT + a.chunk - 1) // a.chunk; log(f"{nchunk} chunks of {a.chunk:,}; resuming at chunk {start}"); timing = []
def score(pairs): 
    f = extract_attention_pair_features(backbone, proteome, pairs, device=dev); return score_pairs_supervised(model=sup, protein_embeddings=proteome.protein_embeddings, pair_features=f, pair_indices=pairs, batch_size=200000, device=dev)
for c in range(start, nchunk):
    k0 = c * a.chunk; k1 = min(NT, k0 + a.chunk); k = np.arange(k0, k1, dtype=np.int64); i, j = untri(k); pairs = np.c_[i, j].astype(np.int64); sync(); t = time.time()
    s = score(pairs); sync(); dt = time.time() - t; assert len(s) == len(pairs) and np.isfinite(s).all(); S[k0:k1] = s; S.flush(); open(pp, "w").write(str(c + 1)); timing.append(dt)
    log(f"chunk {c + 1}/{nchunk}: {len(pairs):,} pairs in {dt:.1f}s = {len(pairs) / dt:,.0f} pairs/s; ETA {(nchunk - c - 1) * np.mean(timing) / 60:.0f} min")
# check: 2,000 random pairs through an explicit pair list must match the stored scores
rs = np.random.RandomState(7); kk = np.sort(rs.choice(NT, 2000, replace=False)); i, j = untri(kk); s2 = score(np.c_[i, j].astype(np.int64)); d = np.abs(s2 - S[kk]); log(f"CHECK explicit-list vs stored: max |diff| {d.max():.3e}, mean {d.mean():.3e}")
json.dump({"gpu": torch.cuda.get_device_name(0), "proteins": N, "pairs": NT, "chunk": a.chunk, "embed_forward_s": embed_s, "chunk_s": timing, "scoring_s_total": float(sum(timing)), "total_s": time.time() - T0, "pairs_per_s_scoring": NT / max(sum(timing), 1e-9), "check_max_abs_diff": float(d.max())}, open(os.path.join(a.out, "timing.json"), "w"), indent=1); log("DONE")
