"""Contact-verified PDB subset, stage 1 (human). For every human PDB-tier positive (variant A, level L0 of bern_univ/out3/human/cand_scores.tsv.gz) that has a PDB entry with <= MAXCH chains
holding both proteins (SIFTS chain->UniProt map), download that entry from RCSB (public PDB-format file, 4 threads, resumable, nothing kept but the contact table) and count residues of each protein within 4.5 A
of the partner. Output pdb_contact_full.tsv: i, j, entry, nch, chainA, chainB, na, nb (one row per pair x entry x chain pair with any atom contact; pairs with no contact in a fetched entry get na=nb=0).
direct = na >= 3 and nb >= 3 (decided downstream). Run: python3 -I pdb_contact_full.py [MAXCH=8]"""
import sys, os, gzip, collections, time, urllib.request, concurrent.futures as cf, numpy as np, pandas as pd
from scipy.spatial import cKDTree
H = "<ROOT>/human_af_screen/"; MAXCH = int(sys.argv[1]) if len(sys.argv) > 1 else 8; OUT = H + "fig2/pdb_contact_full.tsv"; DONE = H + "fig2/pdb_contact_full.done"
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]
B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t", usecols=["i", "j", "label", "tier", "level", "variant"])
P = B[(B.variant == "A") & (B.level == "L0") & (B.tier == "pdb") & (B.label == 1)].drop_duplicates(["i", "j"]).reset_index(drop=True)
need = set(P.i.map(lambda x: ids[x])) | set(P.j.map(lambda x: ids[x])); ent = collections.defaultdict(lambda: collections.defaultdict(set)); nch = collections.defaultdict(set)
with gzip.open(H + "pdb/pdb_chain_uniprot.tsv.gz", "rt") as f:
    next(f); next(f)
    for l in f:
        p = l.rstrip("\n").split("\t")
        if len(p) > 2:
            nch[p[0]].add(p[1])
            if p[2] in need: ent[p[2]][p[0]].add(p[1])
work = collections.defaultdict(list)   # entry -> [(i, j, a, b)]
for r in P.itertuples():
    a, b = ids[r.i], ids[r.j]
    for e in set(ent[a]) & set(ent[b]):
        if len(nch[e]) <= MAXCH: work[e].append((r.i, r.j, a, b))
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set(); todo = [e for e in sorted(work) if e not in done]
print(f"pairs {len(P)}; entries <= {MAXCH} chains {len(work)}; already done {len(done)}; to do {len(todo)}", flush=True)
def fetch(e):
    for k in range(3):
        try:
            t = urllib.request.urlopen(f"https://files.rcsb.org/download/{e.upper()}.pdb", timeout=60).read().decode(); return t
        except urllib.error.HTTPError as ex:
            if ex.code == 404: return None
        except Exception: time.sleep(2 + 3 * k)
    return "ERR"
def one(e):
    t = fetch(e)
    if t is None or t == "ERR": return e, [(i, j, e, len(nch[e]), "", "", -1 if t is None else -2, 0) for i, j, a, b in work[e]]
    want = set(); [want.update(ent[a][e] | ent[b][e]) for i, j, a, b in work[e]]; at = collections.defaultdict(list)
    for l in t.splitlines():
        if l.startswith("ATOM") and l[21] in want: at[l[21]].append((int(l[22:26]), l[26], float(l[30:38]), float(l[38:46]), float(l[46:54])))
    A = {c: (np.array([x[2:] for x in v]), [(x[0], x[1]) for x in v]) for c, v in at.items()}; rows = []
    for i, j, a, b in work[e]:
        best = (0, 0, "", "")
        for x in ent[a][e]:
            for y in ent[b][e]:
                if x in A and y in A and x != y:
                    pr = cKDTree(A[x][0]).query_ball_tree(cKDTree(A[y][0]), 4.5); ia = [k for k, v in enumerate(pr) if v]; ib = {m for v in pr for m in v}
                    na = len({A[x][1][k] for k in ia}); nb = len({A[y][1][k] for k in ib})
                    if na + nb > best[0] + best[1]: best = (na, nb, x, y)
        rows.append((i, j, e, len(nch[e]), best[2], best[3], best[0], best[1]))
    return e, rows
if not os.path.exists(OUT): open(OUT, "w").write("i\tj\tentry\tnch\tchainA\tchainB\tna\tnb\n")
t0 = time.time(); n = 0
with cf.ThreadPoolExecutor(4) as ex, open(OUT, "a") as fo, open(DONE, "a") as fd:
    for e, rows in ex.map(one, todo):
        for r in rows: fo.write("\t".join(map(str, r)) + "\n")
        fd.write(e + "\n"); n += 1
        if n % 200 == 0: fo.flush(); fd.flush(); print(f"{n}/{len(todo)} entries, {(time.time() - t0) / 60:.1f} min", flush=True)
print("DONE", n, "entries", flush=True)
