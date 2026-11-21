#!/usr/bin/env python3
"""PanPPI (one seed) for the multi-species screen. Scores (i) ALL pairs among a fixed random subset of M proteins of the species (the sample whose score distribution gives the rank fractions) with the all-vs-all kernel
and (ii) the AFM-confident pairs with the pairs scorer, 32-bit embeddings, same maths/calibration as the manuscript. A self-check compares the two scorers on 200 random subset pairs first (abort if > 5e-3 logits).
usage: newsp_panppi.py SPECIES RUN_DIR TAG [--smoke]   (cwd = FastPPI2; PYTHONPATH includes panppi_tool)"""
import os, sys, time, numpy as np, pandas as pd, torch
from panppi.score import read_fasta, load_model, embed, score_pairs, AllVsAll
sp, run, tag = sys.argv[1:4]; smoke = "--smoke" in sys.argv; D = f"data/newsp/{sp}"; O = f"analysis_out/newsp/{sp}"; os.makedirs(O, exist_ok=True); t0 = time.time()
log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {sp} {tag}: {m}", flush=True)
seqs = read_fasta(D + "/needed.fasta"); sub = [l.strip() for l in open(D + "/subset_ids.txt") if l.strip()]; afm = pd.read_csv(D + "/afm_pass_pairs.csv")
if smoke: sub = sub[:200]; afm = afm.head(200)
need = sorted(set(sub) | set(afm.a) | set(afm.b)); dev = torch.device("cuda")
model, st = load_model(f"{run}/output/checkpoints/best.pt", f"{run}/calibration.json", "ext/esm2_650M_hf", dev); log(f"model loaded; embedding {len(need):,} proteins")
cache = embed(model, seqs, need, dev); log("embedded")
av = AllVsAll(model, st, cache, sub, dev, batch=256); N = len(sub)
rs = np.random.RandomState(1); ii, jj = rs.randint(0, N, 200), rs.randint(0, N, 200); k = ii != jj; ii, jj = ii[k], jj[k]
P = pd.DataFrame({"protein_a": [sub[x] for x in ii], "protein_b": [sub[x] for x in jj]}); g_ref, m_ref = score_pairs(model, st, P, {a: cache[a].to(dev) for a in set(P.protein_a) | set(P.protein_b)}, dev)
dmax = 0.0
for q in range(len(ii)):
    g, m = av.anchor(int(ii[q]), np.array([jj[q]])); dmax = max(dmax, abs(0.5 * (g[0] + m[0]) - 0.5 * (g_ref[q] + m_ref[q])))
log(f"self-check: max |blend diff| between the two scorers {dmax:.2e}")
if dmax > 5e-3: print("SELF-CHECK FAILED", flush=True); sys.exit(3)
out = np.empty(N * (N - 1) // 2, np.float32); pos = 0
for i in range(N - 1):
    g, m = av.anchor(i, np.arange(i + 1, N)); out[pos:pos + len(g)] = 0.5 * (g + m); pos += len(g)
    if (i + 1) % 1000 == 0: log(f"anchor {i + 1}/{N}")
assert pos == len(out); log(f"scored {len(out):,} subset pairs")
ga, ma = score_pairs(model, st, pd.DataFrame({"protein_a": afm.a, "protein_b": afm.b}), {a: cache[a].to(dev) for a in need}, dev); log(f"scored {len(afm):,} AFM-confident pairs")
fn = f"{O}/panppi_{tag}{'_smoke' if smoke else ''}.npz"; np.savez(fn + ".tmp.npz", subset_blend=out, afm_blend=(0.5 * (ga + ma)).astype(np.float32), afm_a=afm.a.to_numpy(), afm_b=afm.b.to_numpy(), subset_ids=np.array(sub)); os.replace(fn + ".tmp.npz", fn)
log(f"DONE {fn}; peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
