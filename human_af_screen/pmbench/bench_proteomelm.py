#!/usr/bin/env python3
"""End-to-end RUNTIME of ProteomeLM-PPI (released ProteomeLM-S + bundled D-SCRIPT-trained head) on the first N pairs of the 14M-pair benchmark file, repo functions unchanged. Run from ext/ProteomeLM with ext/pm_env python.
Nothing here is a prediction-quality result. usage: bench_proteomelm.py INPUT_DIR OUT_DIR [--pairs 14000000] [--chunk 1000000]"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from proteomelm.ppi.notebook_inference import (load_proteomelm_backbone, prepare_notebook_proteome, load_supervised_ppi_model, score_pair_chunks)
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--pairs", type=int, default=14000000); ap.add_argument("--chunk", type=int, default=1000000)
ap.add_argument("--backbone", default="Bitbol-Lab/ProteomeLM-S"); ap.add_argument("--head", default="data/interactomes/enhanced_ppi_model_dscript.pt"); a = ap.parse_args()
dev = "cuda"; sync = lambda: torch.cuda.synchronize(); os.makedirs(a.out, exist_ok=True); R = {"gpu": torch.cuda.get_device_name(0), "pairs": a.pairs, "chunk": a.chunk}
T0 = time.time(); sync(); t = time.time(); backbone = load_proteomelm_backbone(a.backbone, device=dev); sync(); R["load_backbone_s"] = time.time() - t
cache = os.path.join(a.out, "esmc_cache_fresh.pt"); assert not os.path.exists(cache); t = time.time()
proteome = prepare_notebook_proteome(a.backbone, os.path.join(a.inp, "proteins.fasta"), encoded_genome_file=cache, esm_device="cuda:0", proteomelm_device=dev, model=backbone, reload_if_possible=False); sync(); R["embed_and_forward_s"] = time.time() - t
labels = list(proteome.labels); idx = {l: i for i, l in enumerate(labels)}; R["proteins"] = len(labels); print(f"proteome {len(labels)} proteins: embed+forward {R['embed_and_forward_s']:.0f}s", flush=True)
t = time.time(); sup = load_supervised_ppi_model(a.head, protein_embed_dim=int(proteome.protein_embeddings.shape[1]), pair_feature_dim=1, device=dev, expected_backbone=a.backbone); sync(); R["load_head_s"] = time.time() - t
csv = os.path.join(a.out, "scores.csv"); open(csv, "w").write("protein_a,protein_b,score\n"); chunk_s, write_s, read_s, n_done = [], [], [], 0; reader = pd.read_csv(os.path.join(a.inp, "pairs.csv"), chunksize=a.chunk, dtype=str, usecols=["query", "text"])
t_pairs = time.time()
for ch in reader:
    if n_done >= a.pairs: break
    tr = time.time(); ch = ch.iloc[:a.pairs - n_done]; ia = ch["query"].map(idx).to_numpy(); ib = ch["text"].map(idx).to_numpy(); assert not (np.isnan(ia.astype(float)).any() or np.isnan(ib.astype(float)).any())
    arr = np.c_[np.minimum(ia, ib), np.maximum(ia, ib)].astype(np.int64); read_s.append(time.time() - tr); sync(); tc = time.time()
    res = score_pair_chunks(backbone, proteome, [arr], device=dev, logistic_model=None, supervised_model=sup); sync(); chunk_s.append(time.time() - tc); tw = time.time()
    res[["protein_a", "protein_b", "supervised_score"]].to_csv(csv, mode="a", header=False, index=False); write_s.append(time.time() - tw); n_done += len(ch)
    print(f"  chunk {len(chunk_s)}: {len(ch):,} pairs, score {chunk_s[-1]:.1f}s ({len(ch) / chunk_s[-1]:,.0f} pairs/s), write {write_s[-1]:.1f}s, read+map {read_s[-1]:.1f}s; done {n_done:,}", flush=True)
R["pairs_done"] = n_done; R["score_s_total"] = float(sum(chunk_s)); R["write_s_total"] = float(sum(write_s)); R["read_map_s_total"] = float(sum(read_s)); R["pairs_phase_s"] = time.time() - t_pairs; R["total_s"] = time.time() - T0
R["end_to_end_pairs_per_s"] = n_done / R["total_s"]; R["scoring_pairs_per_s"] = n_done / R["score_s_total"]; R["chunk_rates"] = [a.chunk / s for s in chunk_s]
for N, nm in ((14_000_000, "hours_for_14M"), (188_325_528, "hours_for_188M_scoring_only_plus_fixed")): R[nm] = (R["load_backbone_s"] + R["embed_and_forward_s"] + R["load_head_s"] + N / R["scoring_pairs_per_s"]) / 3600
json.dump(R, open(os.path.join(a.out, "bench_proteomelm_h200.json"), "w"), indent=1); print(json.dumps({k: v for k, v in R.items() if k != "chunk_rates"}, indent=1), flush=True)
