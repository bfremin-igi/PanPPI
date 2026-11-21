#!/usr/bin/env python3
"""Bernett-style degree-balanced versions of the eight cross-species test sets. ALL original positives are kept; negatives are CONSTRUCTED (not the original random ones) so that every protein appears in
the negatives exactly as often as in the positives (1:1 like Bernett's test set), by stub pairing with rejection of self-pairs, duplicates, every original positive and every AlphaFold-DB candidate pair
(known STRING-physical-linked pair) of the species. Pool = the proteins that carry at least one positive. REPS independent negative replicates. Output pairs per species + a feasibility report."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AU
B = "/private/tmp/claude-1618750309/-Users-<user>-oliva-hpc/a067ef54-32be-4e05-a495-9d04f41d958f/scratchpad/"; H = "<ROOT>/human_af_screen/"; OUT = H + "xsp_bern/out/"; os.makedirs(OUT, exist_ok=True); REPS = 2
UNIV = {"yeast": ("yeast", "yeast"), "mouse": ("mouse", "mouse"), "fly": ("fly", "fly"), "worm": ("worm", "worm"), "208964": ("paer", "P_aeruginosa"), "83332": ("mtb", "M_tuberculosis"), "ecoli": (None, None), "224308": (None, None)}
TIER = {"yeast": "yeast", "mouse": "mouse", "fly": "fly", "worm": "worm", "paer": "P_aeruginosa", "mtb": "M_tuberculosis"}
def candidates(sp):
    u = UNIV[sp][0]
    if not u: return set()
    mp = pd.read_csv(H + f"univ/{u}/string_to_uniprot.tsv", sep="\t", header=None, names=["s", "u"], dtype=str); inv = mp.groupby("u").s.apply(list).to_dict()
    T = pd.read_csv(H + f"afdb/tiers_{TIER[u]}.tsv.gz", sep="\t", dtype={"uniprot_ac_1": str, "uniprot_ac_2": str}, usecols=["uniprot_ac_1", "uniprot_ac_2"]); S = set()
    for a, b in zip(T.uniprot_ac_1, T.uniprot_ac_2):
        for x in inv.get(a, ()):
            for y in inv.get(b, ()): S.add((x, y) if x < y else (y, x))
    return S
def balanced(pa, pb, forb, rng, rounds=None):
    """Degree-preserving negative graph: Havel-Hakimi style construction (the node with the largest remaining degree is joined to the eligible nodes with the largest remaining degrees, random tie-breaks), then double-edge swaps that respect the forbidden set. Uniform stub pairing fails on hub-heavy degree sequences, this does not."""
    prot = np.unique(np.r_[pa, pb]); idx = {p: i for i, p in enumerate(prot)}; ia = np.array([idx[x] for x in pa]); ib = np.array([idx[x] for x in pb]); n = len(prot)
    deg = np.bincount(np.r_[ia, ib], minlength=n); rem = deg.astype(np.int64).copy(); forbid = [set() for _ in range(n)]
    for x, y in forb:
        if x in idx and y in idx: forbid[idx[x]].add(idx[y]); forbid[idx[y]].add(idx[x])
    adj = [set() for _ in range(n)]; E = []
    while rem.max() > 0:
        key = rem + rng.random(n) * 0.5; u = int(np.argmax(np.where(rem > 0, key, -1))); need = int(rem[u]); ok = (rem > 0); ok[u] = False
        for v in forbid[u] | adj[u]: ok[v] = False
        cand = np.flatnonzero(ok)
        if len(cand) == 0: rem[u] = 0; continue
        order = cand[np.argsort(-(rem[cand] + rng.random(len(cand)) * 0.5))][:need]
        for v in order: E.append((u, int(v))); adj[u].add(int(v)); adj[int(v)].add(u); rem[v] -= 1
        rem[u] -= len(order)
    left = int(rem.sum()); E = [list(e) for e in E]; m = len(E); sw = 0
    for _ in range(10 * m):
        if m < 2: break
        i, j = rng.integers(0, m, 2)
        if i == j: continue
        a, b = E[i]; c, d = E[j]
        if rng.random() < 0.5: c, d = d, c
        if len({a, b, c, d}) < 4 or d in adj[a] or b in adj[c] or d in forbid[a] or b in forbid[c]: continue
        adj[a].discard(b); adj[b].discard(a); adj[c].discard(d); adj[d].discard(c); adj[a].add(d); adj[d].add(a); adj[c].add(b); adj[b].add(c); E[i] = [a, d]; E[j] = [c, b]; sw += 1
    A = np.array([e[0] for e in E], int); Bb = np.array([e[1] for e in E], int); return prot[A], prot[Bb], left, deg
rng0 = np.random.default_rng(2026); rep = []; allp = []
for sp in UNIV:
    d = pd.read_csv(B + f"unc/pairs_{sp}.tsv.gz", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label"]); P = d[d.label == 1]; pa = P.a.to_numpy(); pb = P.b.to_numpy()
    forb = {(x, y) if x < y else (y, x) for x, y in zip(pa, pb)} | candidates(sp); nneg = []; left = []; dsc = []; frames = [P.assign(rep=-1)]
    for r in range(REPS):
        na, nb, lf, deg = balanced(pa, pb, forb, np.random.default_rng(rng0.integers(1 << 31))); N = pd.DataFrame({"a": na, "b": nb, "label": 0, "rep": r}); frames.append(N); nneg.append(len(N)); left.append(int(lf))
        D = pd.concat([P.assign(rep=-1), N]); dg = pd.concat([D.a, D.b]).value_counts(); hub = np.log1p(D.a.map(dg).to_numpy()) + np.log1p(D.b.map(dg).to_numpy()); dsc.append(AU(D.label.to_numpy(), hub))
    pd.concat(frames).to_csv(OUT + f"{sp}_balanced_pairs.tsv.gz", sep="\t", index=False)
    rep.append(dict(set=sp, positives=len(P), proteins=len(set(pa) | set(pb)), max_pos_degree=int(pd.concat([P.a, P.b]).value_counts().max()), forbidden_candidates=len(candidates(sp)), neg_per_rep=nneg, neg_frac=round(float(np.mean(nneg)) / len(P), 3), stubs_left=left, degree_only_auroc=[round(float(x), 3) for x in dsc]))
    print(rep[-1], flush=True)
json.dump(rep, open(OUT + "feasibility.json", "w"), indent=1)
