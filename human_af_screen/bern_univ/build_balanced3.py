#!/usr/bin/env python3
"""Bernett-style, degree-balanced test sets inside a species universe, scored with every tool by LOOKUP in the finished all-vs-all score files. CPU only.
usage: build_balanced3.py SPECIES   (SPECIES in human ecoli mtb paer yeast fly mouse worm). Env ROOT (working dir), REPS (negative replicates, default 5), OUT.
Positives = truth tier pairs that are AlphaFold-DB candidates: STRING physical >= 700 ('s700'), PDB co-occurrence ('pdb'), AFM-confident ('afm').
Negatives = pairs that are NOT an AFDB candidate (so not in the STRING physical network at any score) drawn so that every protein appears in the negatives exactly as often as in the positives
(v3 = v2 + whole files read into memory + per-cell checkpoint files; Havel-Hakimi construction + degree-preserving edge swaps, rejecting self-pairs, duplicates and candidate pairs; unmatched degree is counted). Levels: L0 all proteins; L2 only proteins with NO homolog (identity >= 0.4, query cov >= 0.5)
in the tools' human training proteins (positives and negatives both restricted to them); for human also L1 = positives whose stratum is 3-5 (pair not seen in training / not a homolog of a training pair).
Pair index k = i*N - i(i+1)/2 + (j-i-1), proteins in sorted id order (= ids.txt)."""
import json, os, sys, time
import numpy as np, pandas as pd
SP = sys.argv[1]; ROOT = os.environ.get("ROOT", "."); os.chdir(ROOT); REPS = int(os.environ.get("REPS", 3)); t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {SP}: {m}", flush=True)
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
R["tri"] = tri(R.i.to_numpy(), R.j.to_numpy()); R = R.drop_duplicates("tri").reset_index(drop=True); FORB = np.sort(R.tri.to_numpy()); _s = np.r_[R.i.to_numpy(), R.j.to_numpy()]; _d = np.r_[R.j.to_numpy(), R.i.to_numpy()]; _o = np.argsort(_s, kind="stable"); FSRC = _s[_o]; FDST = _d[_o]; FPTR = np.searchsorted(FSRC, np.arange(N + 1)); log(f"candidates {len(R):,}: S700 {int((R.string >= 700).sum()):,}, PDB {int(R.pdb.sum()):,}, AFM {int(R.passes.sum()):,}")
H = pd.read_csv(HOM, sep="\t"); H = H.assign(pid=lambda d: d["query"].astype(str).str.split("|").str[-1]); mc = H.set_index("pid").maxid_cov50; mc = mc[~mc.index.duplicated()]
has_hom = np.array([bool(mc.get(p, np.nan) >= 0.4) if p in mc.index else False for p in ids]); log(f"proteins with a training homolog: {int(has_hom.sum()):,} of {N:,}; without: {int((~has_hom).sum()):,}")
# ---- score lookup
G = {s: np.fromfile(f"{PAN[s]}/cal_global.f32", dtype=np.float32) for s in SEEDS}; MX = {s: np.fromfile(f"{PAN[s]}/cal_maxsim.f32", dtype=np.float32) for s in SEEDS}   # v3: read each file ONCE (sequential) instead of random access through a memory map
assert all(len(v) == NT for v in list(G.values()) + list(MX.values())), "score file length differs from N(N-1)/2"; log("score files loaded into memory")
bounds = np.array([N * b // KB for b in range(KB + 1)]); BL = {(p, q): np.load(f"{LG}/bp_{p}_{q}.npy") for p in range(KB) for q in range(p, KB)}; log("LoGoPPI blocks loaded into memory")
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
def balanced_negatives(pi, pj, allowed, rng, rounds=None):
    """Degree-preserving negative graph (v2): Havel-Hakimi construction (largest remaining degree first, joined to the eligible nodes with the largest remaining degrees, random tie-breaks), then double-edge swaps (10 x edges)
    that keep the degree sequence and respect the forbidden set. Uniform stub pairing (v1) fails on heterogeneous degree sequences. Returns (A, B, unmatched stubs = sum(degree) - 2 x edges)."""
    deg = np.bincount(np.r_[pi, pj], minlength=N); nodes = np.flatnonzero(deg > 0); n = len(nodes); loc = -np.ones(N, np.int64); loc[nodes] = np.arange(n); rem = deg[nodes].astype(np.int64).copy(); d0 = int(rem.sum())
    fi = np.searchsorted(FSRC, nodes); forbid = []
    for u in nodes: forbid.append(set(loc[FDST[FPTR[u]:FPTR[u + 1]]][loc[FDST[FPTR[u]:FPTR[u + 1]]] >= 0].tolist()))
    okn = allowed[nodes]; adj = [set() for _ in range(n)]; E = []
    while rem.max() > 0:
        key = rem + rng.random(n) * 0.5; u = int(np.argmax(np.where(rem > 0, key, -1))); need = int(rem[u]); ok = (rem > 0) & okn; ok[u] = False
        for v in forbid[u] | adj[u]: ok[v] = False
        cand = np.flatnonzero(ok)
        if len(cand) == 0: rem[u] = 0; continue
        order = cand[np.argsort(-(rem[cand] + rng.random(len(cand)) * 0.5))][:need]
        for v in order: E.append((u, int(v))); adj[u].add(int(v)); adj[int(v)].add(u); rem[v] -= 1
        rem[u] -= len(order)
    E = [list(e) for e in E]; m = len(E)
    for _ in range(10 * m):
        if m < 2: break
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = E[i]; c, d = E[j]
        if rng.random() < 0.5: c, d = d, c
        if len({a, b, c, d}) < 4 or d in adj[a] or b in adj[c] or d in forbid[a] or b in forbid[c]: continue
        adj[a].discard(b); adj[b].discard(a); adj[c].discard(d); adj[d].discard(c); adj[a].add(d); adj[d].add(a); adj[c].add(b); adj[b].add(c); E[i] = [a, d]; E[j] = [c, b]
    A = nodes[np.array([e[0] for e in E], int)] if E else np.empty(0, int); B = nodes[np.array([e[1] for e in E], int)] if E else np.empty(0, int); return A, B, d0 - 2 * m
O = os.environ.get("OUT", f"analysis_out/bern_univ_v2/{SP}"); os.makedirs(f"{O}/cells", exist_ok=True); metas = []
LEVELS = [("L0", np.ones(N, bool))] + ([("L1", np.ones(N, bool))] if HUMAN else []) + [("L2", ~has_hom)]
for ci, (tier, tmask) in enumerate((("s700", (R.string >= 700).to_numpy()), ("pdb", R.pdb.to_numpy()), ("afm", R.passes.to_numpy()))):
    for li, (lvl, allowed) in enumerate(LEVELS):
        cf = f"{O}/cells/{tier}_{lvl}.tsv.gz"; mf = f"{O}/cells/{tier}_{lvl}.json"
        if os.path.exists(mf): metas.append(json.load(open(mf))); log(f"{tier} {lvl}: cell already done, kept"); continue
        P = R[tmask & allowed[R.i.to_numpy()] & allowed[R.j.to_numpy()]]
        if lvl == "L1": P = P[P.st >= 3]
        if len(P) < 200: m = dict(species=SP, tier=tier, level=lvl, n_pos=int(len(P)), note="too few positives, skipped"); json.dump(m, open(mf, "w")); metas.append(m); log(f"{tier} {lvl}: only {len(P)} positives, skipped"); continue
        pi, pj = P.i.to_numpy(), P.j.to_numpy(); sp_ = score(pi, pj); sp_["label"] = 1; sp_["rep"] = -1; sp_["tier"] = tier; sp_["level"] = lvl; frames = [sp_]
        allowed_neg = allowed if lvl != "L1" else np.ones(N, bool); kept = []
        for r in range(REPS):
            A, B, left = balanced_negatives(pi, pj, allowed_neg, np.random.default_rng([2026, ci, li, r]))
            neg = score(A, B); neg["label"] = 0; neg["rep"] = r; neg["tier"] = tier; neg["level"] = lvl; frames.append(neg); kept.append((len(A), left)); log(f"{tier} {lvl}: replicate {r} built ({len(A):,} negatives)")
        dp = np.bincount(np.r_[pi, pj], minlength=N); m = dict(species=SP, tier=tier, level=lvl, n_pos=int(len(P)), n_proteins=int((dp > 0).sum()), neg_per_rep=[k[0] for k in kept], stubs_left=[k[1] for k in kept])
        pd.concat(frames).to_csv(cf + ".tmp", sep="\t", index=False, float_format="%.6g", compression="gzip"); os.replace(cf + ".tmp", cf); json.dump(m, open(mf, "w")); metas.append(m)
        log(f"{tier} {lvl}: {len(P):,} positives, {int((dp > 0).sum()):,} proteins; negatives per replicate {[k[0] for k in kept]}, unmatched stubs {[k[1] for k in kept]}")
parts = [pd.read_csv(f"{O}/cells/{m['tier']}_{m['level']}.tsv.gz", sep="\t") for m in metas if "note" not in m]
pd.concat(parts).to_csv(f"{O}/balanced_scores.tsv.gz", sep="\t", index=False, float_format="%.6g"); json.dump(metas, open(f"{O}/balanced_meta.json", "w"), indent=1); log("DONE")
