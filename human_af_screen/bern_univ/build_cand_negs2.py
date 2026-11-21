#!/usr/bin/env python3
"""AFM HEAD-TO-HEAD, TRIMMED VARIANT (v2 of build_cand_negs.py: positives are trimmed to the subset whose degree sequence the candidate graph can realise exactly) of build_balanced3.py: positives = STRING>=700 / PDB candidates, negatives = degree-matched LOW-CONFIDENCE AlphaFold-DB candidates (STRING < 400, no PDB; variant B also without AFM-confident pairs) so AFM ipSAE/ipTM exist for every pair. Original docstring: Bernett-style, degree-balanced test sets inside a species universe, scored with every tool by LOOKUP in the finished all-vs-all score files. CPU only.
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
    T = pd.read_csv("data/humanaf/candidates_afm.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}); ia = T.uniprot_ac_1.map(pos); ib = T.uniprot_ac_2.map(pos); ok = (ia.notna() & ib.notna() & (ia != ib)).to_numpy()
    R = pd.DataFrame({"i": ia[ok].astype(int).to_numpy(), "j": ib[ok].astype(int).to_numpy(), "passes": T.passes[ok].to_numpy().astype(bool), "string": T.string[ok].to_numpy(), "pdb": T.pdb[ok].to_numpy().astype(bool), "st": T.st[ok].to_numpy(), "ipsae": T.max_ipSAE[ok].to_numpy(), "iptm": T.ipTM[ok].to_numpy()})
else:
    mp = pd.read_csv(f"data/univ/{SP}/string_to_uniprot.tsv", sep="\t", header=None, names=["s", "u"], dtype=str); inv = mp.groupby("u").s.apply(list).to_dict()
    T = pd.read_csv(f"data/univ/{SP}/tiers.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}); rows = []
    for a, b, ps, st, pdb, ips, itm in zip(T.uniprot_ac_1, T.uniprot_ac_2, T.passes, T.string, T.pdb, T.max_ipSAE, T.ipTM):
        for x in inv.get(a, ()):
            for y in inv.get(b, ()):
                if x != y and x in pos and y in pos: rows.append((pos[x], pos[y], bool(ps), st, bool(pdb), 0, ips, itm))
    R = pd.DataFrame(rows, columns=["i", "j", "passes", "string", "pdb", "st", "ipsae", "iptm"])
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
def cand_negatives(pi, pj, allowed, rng, NEG):
    """Degree-preserving negative graph whose EDGES MUST BE low-confidence AlphaFold-DB candidates (so AFM scores exist for every negative): Havel-Hakimi + degree-preserving swaps inside the allowed graph NEG = (ei, ej)."""
    ei, ej = NEG; deg = np.bincount(np.r_[pi, pj], minlength=N); nodes = np.flatnonzero(deg > 0); n = len(nodes); loc = -np.ones(N, np.int64); loc[nodes] = np.arange(n); rem = deg[nodes].astype(np.int64).copy(); d0 = int(rem.sum())
    m = (loc[ei] >= 0) & (loc[ej] >= 0) & allowed[ei] & allowed[ej]; a_ = loc[ei[m]]; b_ = loc[ej[m]]; s_ = np.r_[a_, b_]; t_ = np.r_[b_, a_]; o = np.argsort(s_, kind="stable"); s_ = s_[o]; t_ = t_[o]; ptr = np.searchsorted(s_, np.arange(n + 1)); nbr = [t_[ptr[u]:ptr[u + 1]] for u in range(n)]
    EDG = set((np.minimum(a_, b_).astype(np.int64) * n + np.maximum(a_, b_)).tolist()); adj = [set() for _ in range(n)]; E = []
    while rem.max() > 0:
        key = rem + rng.random(n) * 0.5; u = int(np.argmax(np.where(rem > 0, key, -1))); need = int(rem[u]); ok = np.zeros(n, bool); ok[nbr[u]] = True; ok &= rem > 0; ok[u] = False
        for v in adj[u]: ok[v] = False
        cand = np.flatnonzero(ok)
        if len(cand) == 0: rem[u] = 0; continue
        order = cand[np.argsort(-(rem[cand] + rng.random(len(cand)) * 0.5))][:need]
        for v in order: E.append((u, int(v))); adj[u].add(int(v)); adj[int(v)].add(u); rem[v] -= 1
        rem[u] -= len(order)
    E = [list(e) for e in E]; mE = len(E)
    for _ in range(10 * mE):
        if mE < 2: break
        i, j = rng.integers(0, mE, 2)
        if i == j: continue
        a, b = E[i]; c, d = E[j]
        if rng.random() < 0.5: c, d = d, c
        if len({a, b, c, d}) < 4 or d in adj[a] or b in adj[c] or (min(a, d) * n + max(a, d)) not in EDG or (min(c, b) * n + max(c, b)) not in EDG: continue
        adj[a].discard(b); adj[b].discard(a); adj[c].discard(d); adj[d].discard(c); adj[a].add(d); adj[d].add(a); adj[c].add(b); adj[b].add(c); E[i] = [a, d]; E[j] = [c, b]
    A = nodes[np.array([e[0] for e in E], int)] if E else np.empty(0, int); Bb = nodes[np.array([e[1] for e in E], int)] if E else np.empty(0, int); return A, Bb, d0 - 2 * mE
def trim_positives(pi, pj, allowed, NEG, rng, probes=6):
    """Largest positive SUBSET (greedy) whose degree sequence the low-confidence-candidate graph can realise. Step 1 (deterministic bound): a protein cannot have more negative partners than it has allowed low-confidence candidate neighbours, so drop positive pairs at proteins whose positive degree exceeds that. Step 2: a few probe constructions; drop positive pairs at proteins whose matched negative degree still falls short (exact deficit)."""
    ei, ej = NEG; keep = np.ones(len(pi), bool); m = allowed[ei] & allowed[ej]; cap = np.bincount(np.r_[ei[m], ej[m]], minlength=N).astype(np.int64)
    def cut(deficit):
        inc = {}
        for e in np.flatnonzero(keep): inc.setdefault(int(pi[e]), []).append(int(e)); inc.setdefault(int(pj[e]), []).append(int(e))
        for u in np.argsort(-deficit):
            if deficit[u] <= 0: break
            cand = [e for e in inc.get(int(u), []) if keep[e]]; cand.sort(key=lambda e: -deficit[pj[e] if pi[e] == u else pi[e]])
            for e in cand:
                if deficit[u] <= 0: break
                keep[e] = False; deficit[pi[e]] -= 1; deficit[pj[e]] -= 1
    for _ in range(4):
        pdg = np.bincount(np.r_[pi[keep], pj[keep]], minlength=N); deficit = pdg - cap
        if deficit.max() <= 0: break
        cut(deficit.astype(np.int64))
    log(f"trim step 1 (allowed-degree bound): {int(keep.sum()):,} of {len(pi):,} positives kept")
    for it in range(probes):
        A, Bq, left = cand_negatives(pi[keep], pj[keep], allowed, rng, NEG)
        if left == 0 or len(A) >= 0.97 * keep.sum(): break
        pdg = np.bincount(np.r_[pi[keep], pj[keep]], minlength=N); ndg = np.bincount(np.r_[A, Bq], minlength=N); cut((pdg - ndg).astype(np.int64))
        log(f"trim probe {it}: {int(keep.sum()):,} positives kept")
    return keep
O = os.environ.get("OUT", f"analysis_out/bern_cand2/{SP}"); os.makedirs(f"{O}/cells", exist_ok=True); metas = []; RI = R.set_index("tri")
LEVELS = [("L0", np.ones(N, bool))] + ([("L1", np.ones(N, bool))] if HUMAN else []) + [("L2", ~has_hom)]
VARIANTS = {"A": ((R.string < 400) & (~R.pdb)).to_numpy(), "B": ((R.string < 400) & (~R.pdb) & (~R.passes)).to_numpy()}   # A: all low-STRING candidates; B: additionally without AFM-confident pairs (favours AFM)
NEGG = {v: (R.i.to_numpy()[m], R.j.to_numpy()[m]) for v, m in VARIANTS.items()}
def add_afm(D):
    k = tri(D.i.to_numpy(), D.j.to_numpy()); X = RI.reindex(k); assert X.ipsae.notna().all(), "pair without AFM scores"; D["ipsae"] = X.ipsae.to_numpy(); D["iptm"] = X.iptm.to_numpy(); D["afm_pass"] = X.passes.to_numpy().astype(int); D["string"] = X.string.to_numpy(); return D
for ci, (tier, tmask) in enumerate((("s700", (R.string >= 700).to_numpy()), ("pdb", R.pdb.to_numpy()))):
    for li, (lvl, allowed) in enumerate(LEVELS):
        for vi, var in enumerate(("A", "B")):
            name = f"{tier}_{lvl}_{var}"; cf = f"{O}/cells/{name}.tsv.gz"; mf = f"{O}/cells/{name}.json"
            if os.path.exists(mf): metas.append(json.load(open(mf))); log(f"{name}: cell already done, kept"); continue
            P = R[tmask & allowed[R.i.to_numpy()] & allowed[R.j.to_numpy()]]
            if lvl == "L1": P = P[P.st >= 3]
            if len(P) < 200: m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(P)), note="too few positives, skipped"); json.dump(m, open(mf, "w")); metas.append(m); log(f"{name}: only {len(P)} positives, skipped"); continue
            pi, pj = P.i.to_numpy(), P.j.to_numpy(); allowed_neg = allowed if lvl != "L1" else np.ones(N, bool); n_all = len(pi)
            keepm = trim_positives(pi, pj, allowed_neg, NEGG[var], np.random.default_rng([2028, ci, li, vi])); pi, pj = pi[keepm], pj[keepm]; log(f"{name}: positives kept after trimming {len(pi):,} of {n_all:,}")
            if len(pi) < 200: m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(pi)), n_pos_total=int(n_all), note="too few positives after trimming, skipped"); json.dump(m, open(mf + f".tmp{os.getpid()}", "w")); os.replace(mf + f".tmp{os.getpid()}", mf); metas.append(m); continue
            sp_ = add_afm(score(pi, pj)); sp_["label"] = 1; sp_["rep"] = -1; frames = [sp_]; kept = []
            for r in range(REPS):
                A, Bq, left = cand_negatives(pi, pj, allowed_neg, np.random.default_rng([2027, ci, li, vi, r]), NEGG[var])
                neg = add_afm(score(A, Bq)); neg["label"] = 0; neg["rep"] = r; frames.append(neg); kept.append((len(A), left)); log(f"{name}: replicate {r} built ({len(A):,} negatives)")
            for f_ in frames: f_["tier"] = tier; f_["level"] = lvl; f_["variant"] = var
            dp = np.bincount(np.r_[pi, pj], minlength=N); m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(pi)), n_pos_total=int(n_all), retained=round(len(pi) / n_all, 3), n_proteins=int((dp > 0).sum()), neg_per_rep=[k[0] for k in kept], stubs_left=[k[1] for k in kept])
            pd.concat(frames).to_csv(cf + f".tmp{os.getpid()}", sep="\t", index=False, float_format="%.6g", compression="gzip"); os.replace(cf + f".tmp{os.getpid()}", cf); json.dump(m, open(mf + f".tmp{os.getpid()}", "w")); os.replace(mf + f".tmp{os.getpid()}", mf); metas.append(m)   # unique temp names: several arrays may run the same cell
            log(f"{name}: {len(pi):,} of {n_all:,} positives; negatives per replicate {[k[0] for k in kept]}, unmatched degree {[k[1] for k in kept]}")
parts = [pd.read_csv(f"{O}/cells/{m['tier']}_{m['level']}_{m['variant']}.tsv.gz", sep="\t") for m in metas if "note" not in m]
parts = parts or [pd.DataFrame({"i": []})]
pd.concat(parts).to_csv(f"{O}/cand_scores.tsv.gz", sep="\t", index=False, float_format="%.6g"); json.dump(metas, open(f"{O}/cand_meta.json", "w"), indent=1); log("DONE")
