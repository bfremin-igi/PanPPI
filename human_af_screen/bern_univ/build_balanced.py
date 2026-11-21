#!/usr/bin/env python3
"""Bernett-style, degree-balanced test sets inside a species universe, scored with every tool by LOOKUP in the finished all-vs-all score files. CPU only.
usage: build_balanced.py SPECIES   (SPECIES in human ecoli mtb paer yeast fly mouse worm). Env ROOT (working dir), REPS (negative replicates, default 5), OUT.
Positives = truth tier pairs that are AlphaFold-DB candidates: STRING physical >= 700 ('s700'), PDB co-occurrence ('pdb'), AFM-confident ('afm').
Negatives = pairs that are NOT an AFDB candidate (so not in the STRING physical network at any score) drawn so that every protein appears in the negatives exactly as often as in the positives
(stub pairing with rejection of self-pairs, duplicates and candidate pairs; unmatched stubs are dropped and counted). Levels: L0 all proteins; L2 only proteins with NO homolog (identity >= 0.4, query cov >= 0.5)
in the tools' human training proteins (positives and negatives both restricted to them); for human also L1 = positives whose stratum is 3-5 (pair not seen in training / not a homolog of a training pair).
Pair index k = i*N - i(i+1)/2 + (j-i-1), proteins in sorted id order (= ids.txt)."""
import json, os, sys, time
import numpy as np, pandas as pd
SP = sys.argv[1]; ROOT = os.environ.get("ROOT", "."); os.chdir(ROOT); REPS = int(os.environ.get("REPS", 5)); t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {SP}: {m}", flush=True)
HUMAN = SP == "human"; SEEDS = ("s42", "s7", "s141")
if HUMAN:
    ids = [l.strip() for l in open("analysis_out/humanaf_allpairs/ids.txt") if l.strip()]; KB = 8; PAN = {"s42": "analysis_out/humanaf_allpairs", "s7": "analysis_out/humanaf_allpairs_s7", "s141": "analysis_out/humanaf_allpairs_s141"}; LG = "analysis_out/humanaf_logoppi_universe"; HOM = "analysis_out/homology_humanaf/maxid.tsv"
else:
    ids = [l.strip() for l in open(f"data/univ/{SP}/ids.txt") if l.strip()]; KB = {"ecoli": 2, "mtb": 2, "paer": 3, "yeast": 4, "fly": 8, "mouse": 8, "worm": 8}.get(SP) or int(os.environ["KB"]); PAN = {s: f"analysis_out/univ_{SP}/pan_{s}" for s in SEEDS}; LG = f"analysis_out/univ_{SP}/logoppi"; HOM = "analysis_out/univ_homology/maxid.tsv"
N = len(ids); NT = N * (N - 1) // 2; pos = {p: i for i, p in enumerate(ids)}; log(f"{N} proteins, {NT:,} pairs")
def tri(i, j): i, j = np.minimum(i, j).astype(np.int64), np.maximum(i, j).astype(np.int64); return i * N - i * (i + 1) // 2 + (j - i - 1)
# ---- candidates (positives' source and the forbidden set)
if HUMAN:
    T = pd.read_csv("data/humanaf/candidates_with_strata.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}); ia = T.uniprot_ac_1.map(pos); ib = T.uniprot_ac_2.map(pos); ok = (ia.notna() & ib.notna() & (ia != ib)).to_numpy()
    R = pd.DataFrame({"i": ia[ok].astype(int).to_numpy(), "j": ib[ok].astype(int).to_numpy(), "passes": T.passes[ok].to_numpy().astype(bool), "string": T.string[ok].to_numpy(), "pdb": T.pdb[ok].to_numpy().astype(bool), "st": T.st[ok].to_numpy()})
else:
    mp = pd.read_csv(f"data/univ/{SP}/string_to_uniprot.tsv", sep="\t", header=None, names=["s", "u"], dtype=str); inv = mp.groupby("u").s.apply(list).to_dict()
    T = pd.read_csv(f"data/univ/{SP}/tiers.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}); rows = []
    for a, b, ps, st, pdb in zip(T.uniprot_ac_1, T.uniprot_ac_2, T.passes, T.string, T.pdb):
        for x in inv.get(a, ()):
            for y in inv.get(b, ()):
                if x != y and x in pos and y in pos: rows.append((pos[x], pos[y], bool(ps), st, bool(pdb), 0))
    R = pd.DataFrame(rows, columns=["i", "j", "passes", "string", "pdb", "st"])
R["tri"] = tri(R.i.to_numpy(), R.j.to_numpy()); R = R.drop_duplicates("tri").reset_index(drop=True); FORB = np.sort(R.tri.to_numpy()); log(f"candidates {len(R):,}: S700 {int((R.string >= 700).sum()):,}, PDB {int(R.pdb.sum()):,}, AFM {int(R.passes.sum()):,}")
H = pd.read_csv(HOM, sep="\t"); H = H.assign(pid=lambda d: d["query"].astype(str).str.split("|").str[-1]); mc = H.set_index("pid").maxid_cov50; mc = mc[~mc.index.duplicated()]
has_hom = np.array([bool(mc.get(p, np.nan) >= 0.4) if p in mc.index else False for p in ids]); log(f"proteins with a training homolog: {int(has_hom.sum()):,} of {N:,}; without: {int((~has_hom).sum()):,}")
# ---- score lookup
G = {s: np.memmap(f"{PAN[s]}/cal_global.f32", dtype=np.float32, mode="r", shape=(NT,)) for s in SEEDS}; MX = {s: np.memmap(f"{PAN[s]}/cal_maxsim.f32", dtype=np.float32, mode="r", shape=(NT,)) for s in SEEDS}
bounds = np.array([N * b // KB for b in range(KB + 1)]); BL = {(p, q): np.load(f"{LG}/bp_{p}_{q}.npy", mmap_mode="r") for p in range(KB) for q in range(p, KB)}
def lg_lookup(i, j):
    i, j = np.minimum(i, j), np.maximum(i, j); bp = np.searchsorted(bounds, i, side="right") - 1; bq = np.searchsorted(bounds, j, side="right") - 1; out = np.full(len(i), np.nan, np.float32)
    for (p, q), arr in BL.items():
        m = (bp == p) & (bq == q)
        if not m.any(): continue
        ii, jj = i[m], j[m]
        if p == q: n = bounds[p + 1] - bounds[p]; u = ii - bounds[p]; v = jj - bounds[p]; idx = u * n - u * (u + 1) // 2 + (v - u - 1)
        else: idx = (ii - bounds[p]) * (bounds[q + 1] - bounds[q]) + (jj - bounds[q])
        out[m] = arr[idx]
    assert np.isfinite(out).all(); return out
def score(i, j):
    k = tri(i, j); D = {"i": np.minimum(i, j), "j": np.maximum(i, j)}
    for s in SEEDS: D[f"g_{s}"] = G[s][k]; D[f"m_{s}"] = MX[s][k]
    D["lg"] = lg_lookup(i, j); return pd.DataFrame(D)
# ---- degree-balanced negatives
def balanced_negatives(pi, pj, allowed, rng, rounds=200):
    deg = np.bincount(np.r_[pi, pj], minlength=N); stubs = np.repeat(np.arange(N), deg); acc = np.empty(0, np.int64); accA = []; accB = []; seen = np.sort(np.r_[FORB])
    for _ in range(rounds):
        if len(stubs) < 2: break
        stubs = rng.permutation(stubs); n2 = len(stubs) // 2; a = stubs[:n2]; b = stubs[n2:2 * n2]; rest = stubs[2 * n2:]; k = tri(a, b)
        bad = (a == b) | (~allowed[a]) | (~allowed[b]); bad |= np.isin(k, seen, assume_unique=False)
        _, first = np.unique(k, return_index=True); dup = np.ones(len(k), bool); dup[first] = False; bad |= dup
        good = ~bad; accA.append(a[good]); accB.append(b[good]); seen = np.sort(np.r_[seen, k[good]]); stubs = np.r_[a[bad], b[bad], rest]
    A = np.concatenate(accA) if accA else np.empty(0, int); B = np.concatenate(accB) if accB else np.empty(0, int); return A, B, len(stubs)
rng0 = np.random.default_rng(2026); out = []; meta = []
LEVELS = [("L0", np.ones(N, bool))] + ([("L1", np.ones(N, bool))] if HUMAN else []) + [("L2", ~has_hom)]
for tier, tmask in (("s700", (R.string >= 700).to_numpy()), ("pdb", R.pdb.to_numpy()), ("afm", R.passes.to_numpy())):
    for lvl, allowed in LEVELS:
        P = R[tmask & allowed[R.i.to_numpy()] & allowed[R.j.to_numpy()]]
        if lvl == "L1": P = P[P.st >= 3]
        if len(P) < 200: meta.append(dict(species=SP, tier=tier, level=lvl, n_pos=int(len(P)), note="too few positives, skipped")); log(f"{tier} {lvl}: only {len(P)} positives, skipped"); continue
        pi, pj = P.i.to_numpy(), P.j.to_numpy(); sp_ = score(pi, pj); sp_["label"] = 1; sp_["rep"] = -1; sp_["tier"] = tier; sp_["level"] = lvl; out.append(sp_)
        allowed_neg = allowed if lvl != "L1" else np.ones(N, bool); kept = []
        for r in range(REPS):
            A, B, left = balanced_negatives(pi, pj, allowed_neg, np.random.default_rng(rng0.integers(1 << 31)))
            neg = score(A, B); neg["label"] = 0; neg["rep"] = r; neg["tier"] = tier; neg["level"] = lvl; out.append(neg); kept.append((len(A), left))
        dp = np.bincount(np.r_[pi, pj], minlength=N); meta.append(dict(species=SP, tier=tier, level=lvl, n_pos=int(len(P)), n_proteins=int((dp > 0).sum()), neg_per_rep=[k[0] for k in kept], stubs_left=[k[1] for k in kept]))
        log(f"{tier} {lvl}: {len(P):,} positives, {int((dp > 0).sum()):,} proteins; negatives per replicate {[k[0] for k in kept]}, unmatched stubs {[k[1] for k in kept]}")
os.makedirs(os.environ.get("OUT", f"analysis_out/bern_univ/{SP}"), exist_ok=True); O = os.environ.get("OUT", f"analysis_out/bern_univ/{SP}")
pd.concat(out).to_csv(f"{O}/balanced_scores.tsv.gz", sep="\t", index=False, float_format="%.6g"); json.dump(meta, open(f"{O}/balanced_meta.json", "w"), indent=1); log("DONE")
