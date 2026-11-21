#!/usr/bin/env python3
"""PanPPI (one seed) on the unique cell pairs of a species (32-bit embeddings, cache on the CPU, pairs scorer). usage: newsp_cells_panppi.py SPECIES RUN_DIR TAG [--smoke]. Output analysis_out/newsp/SP/cells_panppi_TAG.csv (protein_a, protein_b, g, m)."""
import os, sys, time, numpy as np, pandas as pd, torch
from panppi.score import read_fasta, load_model, embed, score_pairs
sp, run, tag = sys.argv[1:4]; smoke = "--smoke" in sys.argv; D = f"data/newsp/{sp}"; O = f"analysis_out/newsp/{sp}"; t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {sp} {tag}: {m}", flush=True)
P = pd.read_csv(f"{O}/cell_pairs.csv", dtype=str); P = P.head(300) if smoke else P; seqs = read_fasta(f"{D}/cand_proteins.fasta"); need = sorted(set(P.protein_a) | set(P.protein_b)); dev = torch.device("cuda")
model, st = load_model(f"{run}/output/checkpoints/best.pt", f"{run}/calibration.json", "ext/esm2_650M_hf", dev); log(f"{len(P):,} pairs, {len(need):,} proteins to embed"); cache = embed(model, seqs, need, dev); log("embedded")
g, m = score_pairs(model, st, P, cache, dev, batch=256, chunk=100000); log("scored"); P = P.assign(g=g, m=m); fn = f"{O}/cells_panppi_{tag}{'_smoke' if smoke else ''}.csv"; P.to_csv(fn + ".tmp", index=False, float_format="%.6g"); os.replace(fn + ".tmp", fn); log(f"DONE {fn}")
