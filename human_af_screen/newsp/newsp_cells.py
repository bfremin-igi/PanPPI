#!/usr/bin/env python3
"""Degree-matched cells for the multi-species screen: the construction of bern_univ/build_cand_negs2.py (trimmed variant A; tier s700; levels L0 and L2; three negative replicates), ported to species without universe scoring.
Positives = AFDB candidates with STRING v12 physical score >= 700; negatives = degree-matched pairs drawn from the low-confidence candidates (STRING < 400, no PDB co-membership) so AFM scores exist for every pair; positives are trimmed to the
subset whose degree sequence that graph can realise; L2 = only proteins with no human-training homolog (MMseqs2 identity >= 0.4, query coverage >= 0.5). Functions cand_negatives / trim_positives are copied unchanged from build_cand_negs2.py.
usage: newsp_cells.py SPECIES   (cwd = FastPPI2). Writes analysis_out/newsp/SP/cells/*.tsv.gz, cell_meta.json and cell_pairs.csv (unique pairs to score)."""
import json, os, sys, time
import numpy as np, pandas as pd
SP = sys.argv[1]; REPS = 3; t0 = time.time(); log = lambda m: print(f"[{(time.time() - t0) / 60:5.1f} min] {SP}: {m}", flush=True)
D = f"data/newsp/{SP}"; O = f"analysis_out/newsp/{SP}"; os.makedirs(f"{O}/cells", exist_ok=True)
Tr = pd.read_csv(f"{D}/tiers.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}); ids = sorted(set(Tr.uniprot_ac_1) | set(Tr.uniprot_ac_2)); pos = {p: i for i, p in enumerate(ids)}; N = len(ids)
R = pd.DataFrame({"i": Tr.uniprot_ac_1.map(pos).to_numpy(), "j": Tr.uniprot_ac_2.map(pos).to_numpy(), "passes": Tr.passes.to_numpy().astype(bool), "string": Tr.string.to_numpy(), "pdb": Tr.pdb.to_numpy().astype(bool), "ipsae": Tr.max_ipSAE.to_numpy(), "iptm": Tr.ipTM.to_numpy()})
a_, b_ = np.minimum(R.i, R.j), np.maximum(R.i, R.j); R["i"], R["j"] = a_.astype(int), b_.astype(int); R = R.drop_duplicates(["i", "j"]).reset_index(drop=True); R["key"] = R.i.to_numpy().astype(np.int64) * N + R.j.to_numpy(); RI = R.set_index("key")
H = pd.read_csv(f"{O}/homology/maxid.tsv", sep="\t"); H = H.assign(pid=lambda d: d["query"].astype(str).str.split("|").str[-1]); mc = H.set_index("pid").maxid_cov50; mc = mc[~mc.index.duplicated()]
has_hom = np.array([bool(mc.get(p, np.nan) >= 0.4) if p in mc.index else False for p in ids]); log(f"{N} proteins, {len(R):,} candidates; proteins with a training homolog: {int(has_hom.sum()):,}; without: {int((~has_hom).sum()):,}")
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
metas = []; LEVELS = [("L0", np.ones(N, bool)), ("L2", ~has_hom)]; var = "A"; neg_mask = ((R.string < 400) & (~R.pdb)).to_numpy(); NEG = (R.i.to_numpy()[neg_mask], R.j.to_numpy()[neg_mask])
def add_afm(i, j):
    X = RI.reindex(np.minimum(i, j).astype(np.int64) * N + np.maximum(i, j)); assert X.ipsae.notna().all(), "pair without AFM scores"
    return pd.DataFrame({"i": np.minimum(i, j), "j": np.maximum(i, j), "a": [ids[x] for x in np.minimum(i, j)], "b": [ids[x] for x in np.maximum(i, j)], "ipsae": X.ipsae.to_numpy(), "iptm": X.iptm.to_numpy(), "afm_pass": X.passes.to_numpy().astype(int), "string": X.string.to_numpy()})
tier, tmask = "s700", (R.string >= 700).to_numpy()
for li, (lvl, allowed) in enumerate(LEVELS):
    name = f"{tier}_{lvl}_{var}"; P = R[tmask & allowed[R.i.to_numpy()] & allowed[R.j.to_numpy()]]
    if len(P) < 200: m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(P)), note="too few positives, skipped"); metas.append(m); log(f"{name}: only {len(P)} positives, skipped"); continue
    pi, pj = P.i.to_numpy(), P.j.to_numpy(); n_all = len(pi); keepm = trim_positives(pi, pj, allowed, NEG, np.random.default_rng([2028, 0, li, 0])); pi, pj = pi[keepm], pj[keepm]; log(f"{name}: positives kept after trimming {len(pi):,} of {n_all:,}")
    if len(pi) < 200: m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(pi)), n_pos_total=int(n_all), note="too few positives after trimming, skipped"); metas.append(m); continue
    sp_ = add_afm(pi, pj); sp_["label"] = 1; sp_["rep"] = -1; frames = [sp_]; kept = []
    for r in range(REPS):
        A, Bq, left = cand_negatives(pi, pj, allowed, np.random.default_rng([2027, 0, li, 0, r]), NEG); neg = add_afm(A, Bq); neg["label"] = 0; neg["rep"] = r; frames.append(neg); kept.append((len(A), left)); log(f"{name}: replicate {r} built ({len(A):,} negatives)")
    for f_ in frames: f_["tier"] = tier; f_["level"] = lvl; f_["variant"] = var
    dp = np.bincount(np.r_[pi, pj], minlength=N); m = dict(species=SP, tier=tier, level=lvl, variant=var, n_pos=int(len(pi)), n_pos_total=int(n_all), retained=round(len(pi) / n_all, 3), n_proteins=int((dp > 0).sum()), neg_per_rep=[k[0] for k in kept], stubs_left=[k[1] for k in kept]); metas.append(m)
    pd.concat(frames).to_csv(f"{O}/cells/{name}.tsv.gz", sep="\t", index=False, float_format="%.6g", compression="gzip"); log(f"{name}: {len(pi):,} of {n_all:,} positives; negatives per replicate {[k[0] for k in kept]}, unmatched degree {[k[1] for k in kept]}")
json.dump(metas, open(f"{O}/cell_meta.json", "w"), indent=1)
# ---- no-trimming sensitivity set: ALL s700 positives (L0) + a uniform sample of the low-confidence candidate pool (negatives to be reweighted by degree, as for Fig 2d)
rng = np.random.default_rng(77); posall = R[tmask]; negpool = R[neg_mask]; nn = min(len(negpool), 300000); ns = negpool.iloc[rng.choice(len(negpool), nn, replace=False)]
RW = pd.concat([add_afm(posall.i.to_numpy(), posall.j.to_numpy()).assign(label=1), add_afm(ns.i.to_numpy(), ns.j.to_numpy()).assign(label=0)]); RW.to_csv(f"{O}/reweight_set.tsv.gz", sep="\t", index=False, float_format="%.6g", compression="gzip"); log(f"reweight set: {len(posall):,} positives, {nn:,} sampled negatives of {len(negpool):,}")
U = pd.concat([pd.read_csv(f"{O}/cells/{m['tier']}_{m['level']}_{m['variant']}.tsv.gz", sep="\t", usecols=["a", "b"]) for m in metas if "note" not in m] + [RW[["a", "b"]]]).drop_duplicates(); U.rename(columns={"a": "protein_a", "b": "protein_b"}).to_csv(f"{O}/cell_pairs.csv", index=False); log(f"DONE; {len(U):,} unique pairs to score")
