#!/usr/bin/env python3
"""BERNETT VERSION of analysis/uncert_features.py (same feature definitions, Bernett split): sets = Bernett test + the two Bernett validation halves; training proteins = Bernett training proteins.
Extra uncertainty / evidence features for the finished E3a run (inference only, no training, no GP):
 per PAIR  : deterministic raw global g and MaxSim m, MC-dropout (T passes of the head's own dropout p=0.1 on the cached projected residues) mean/variance of g and of m,
             MaxSim evidence statistics: directional scores sa, sb, mean gap between each residue's best and second-best match (gap_a, gap_b), spread of the per-residue best matches (sd_a, sd_b)
 per PROTEIN: the pooled projected embedding (512-d, mean over residues), saved fp16, for the 8 original test sets, the new bacterial/archaeal sets and the human TRAINING proteins
             (label-free protein-level familiarity is computed OFFLINE from these: distance to the nearest human training proteins).
Same model construction / checkpoint as the post-hoc runs; encoder in bf16 autocast, head in fp32. NOTHING is a result: it only writes features.
usage: uncert_features.py RUN_DIR OUT_DIR [--limit N] [--T 8]"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
import main_e3 as M


def read_fasta(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): k = l[1:].split()[0]; d[k] = ""
        elif l: d[k] += l.rstrip("*")
    return d


def organism_chunks(df, max_pairs=20000, max_prot=12000):
    org = df["a"].str.split(".").str[0].to_numpy(); order = np.argsort(org, kind="stable"); a_, b_ = df["a"].to_numpy(), df["b"].to_numpy(); chunks, cur, prots = [], [], set()
    for i in order:
        new = {a_[i], b_[i]} - prots
        if cur and (len(cur) >= max_pairs or len(prots) + len(new) > max_prot):
            chunks.append(np.array(cur)); cur, prots = [], set(); new = {a_[i], b_[i]}
        cur.append(i); prots |= new
    if cur: chunks.append(np.array(cur))
    return chunks


def pair_batch(ea, eb, dev):
    d = ea[0].shape[1]; la, lb = [e.shape[0] for e in ea], [e.shape[0] for e in eb]
    xa = torch.zeros(len(ea), max(la), d, device=dev); xb = torch.zeros(len(eb), max(lb), d, device=dev)
    ma = torch.zeros(len(ea), max(la), dtype=torch.bool, device=dev); mb = torch.zeros(len(eb), max(lb), dtype=torch.bool, device=dev)
    for j, (p, q) in enumerate(zip(ea, eb)):
        xa[j, :p.shape[0]], ma[j, :p.shape[0]] = p.float(), True; xb[j, :q.shape[0]], mb[j, :q.shape[0]] = q.float(), True
    return xa, ma, xb, mb


@torch.no_grad()
def evidence(xa, ma, xb, mb):
    sim = torch.bmm(xa, xb.transpose(1, 2)); neg = torch.finfo(sim.dtype).min
    sa_m = sim.masked_fill(~mb[:, None, :], neg); sb_m = sim.masked_fill(~ma[:, :, None], neg).transpose(1, 2)
    def side(s, mq, mt):
        k = min(2, s.shape[2]); top = s.topk(k, dim=2).values; t1 = top[..., 0]; t2 = top[..., 1] if k == 2 else t1
        n = mq.sum(1).clamp_min(1); mean = (t1 * mq).sum(1) / n; gap = ((t1 - t2) * mq).sum(1) / n; var = (((t1 - mean[:, None]) ** 2) * mq).sum(1) / n
        return mean, gap, var.sqrt()
    a_mean, a_gap, a_sd = side(sa_m, ma, mb); b_mean, b_gap, b_sd = side(sb_m, mb, ma)
    return a_mean, b_mean, a_gap, b_gap, a_sd, b_sd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--esm-dir", default="ext/esm2_650M_hf"); a = ap.parse_args()
    dev = torch.device("cuda"); amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    model = M.Model(os.path.abspath(a.esm_dir), 1280, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=141, use_res_gp=False).to(dev)
    sd = torch.load(f"{a.run}/output/checkpoints/best.pt", map_location="cpu", weights_only=False); model.load_state_dict(sd["model"] if "model" in sd else sd); model.eval()
    p_drop = float(model.head.drop.p) if hasattr(model.head, "drop") else 0.1; print("model loaded %.1f s; dropout p = %.2f, T = %d" % (time.time() - t0, p_drop, a.T), flush=True)
    D = "data/bernett/data"
    sets = [("bernett", f"{D}/test/bernett_pairs.csv", f"{D}/test/bernett.fasta"), ("val_selection", f"{D}/human_validation_selection.csv", f"{D}/proteins.fasta"), ("val_calibration", f"{D}/human_validation_calibration.csv", f"{D}/proteins.fasta")]
    # ---- human training proteins (pooled embeddings only)
    allp = read_fasta(f"{D}/proteins.fasta"); tr = pd.read_csv(f"{D}/human_train_clean.csv", dtype=str); ids = sorted(set(tr["query"]) | set(tr["text"]))
    if a.limit: ids = ids[:a.limit]
    pooled_ids, pooled = [], []
    for i in range(0, len(ids), 12000):
        sub = ids[i:i + 12000]; cache = M.embed_cache(model, set(sub), allp, dev, amp)
        for k in sub: pooled_ids.append(k); pooled.append(cache[k].float().mean(0).half().cpu().numpy())
        del cache; torch.cuda.empty_cache()
    np.save(f"{a.out}/prot_train.npy", np.stack(pooled)); pd.Series(pooled_ids).to_csv(f"{a.out}/prot_train.ids", index=False, header=False); print("human train proteins %d, %.1f min" % (len(pooled_ids), (time.time() - t0) / 60), flush=True)
    for tag, pairs_csv, fasta in sets:
        seqs = read_fasta(fasta); df = pd.read_csv(pairs_csv, dtype=str).rename(columns={"query": "a", "text": "b"}); df["label"] = df["label"].astype(int)
        if a.limit: df = df.iloc[:a.limit].copy()
        n = len(df); cols = ["g", "m", "g_mc_mean", "g_mc_var", "m_mc_mean", "m_mc_var", "sa", "sb", "gap_a", "gap_b", "sd_a", "sd_b"]; out = {k: np.zeros(n, dtype=np.float32) for k in cols}
        pids, pvec = {}, {}
        for ci, ix in enumerate(organism_chunks(df)):
            sub = df.iloc[ix]; cache = M.embed_cache(model, set(sub.a) | set(sub.b), seqs, dev, amp)
            for k, x in cache.items():
                if k not in pvec: pvec[k] = x.float().mean(0).half().cpu().numpy()
            A, Bv = sub.a.values, sub.b.values
            with torch.no_grad():
                for i in range(0, len(sub), 64):
                    ea, eb = [cache[k] for k in A[i:i + 64]], [cache[k] for k in Bv[i:i + 64]]; xa, ma, xb, mb = pair_batch(ea, eb, dev); rows = ix[i:i + 64]
                    per = model.head.score_projected(xa, ma, xb, mb, with_maxsim=True); out["g"][rows] = per["global"].float().cpu().numpy(); out["m"][rows] = per["maxsim"].float().cpu().numpy()
                    ev = evidence(xa, ma, xb, mb)
                    for k, v in zip(("sa", "sb", "gap_a", "gap_b", "sd_a", "sd_b"), ev): out[k][rows] = v.float().cpu().numpy()
                    G, Mx = [], []
                    for t in range(a.T):
                        pt = model.head.score_projected(F.dropout(xa, p_drop, True), ma, F.dropout(xb, p_drop, True), mb, with_maxsim=True); G.append(pt["global"].float()); Mx.append(pt["maxsim"].float())
                    G, Mx = torch.stack(G), torch.stack(Mx)
                    out["g_mc_mean"][rows] = G.mean(0).cpu().numpy(); out["g_mc_var"][rows] = G.var(0).cpu().numpy(); out["m_mc_mean"][rows] = Mx.mean(0).cpu().numpy(); out["m_mc_var"][rows] = Mx.var(0).cpu().numpy()
            del cache; torch.cuda.empty_cache(); print(f"  {tag} chunk {ci}: {len(ix):,} pairs ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res = df[["a", "b", "label"]].copy()
        for k in cols: res[k] = out[k]
        res.to_csv(f"{a.out}/pairs_{tag}.tsv.gz", sep="\t", index=False)
        keys = sorted(pvec); np.save(f"{a.out}/prot_{tag}.npy", np.stack([pvec[k] for k in keys])); pd.Series(keys).to_csv(f"{a.out}/prot_{tag}.ids", index=False, header=False)
        print(f"{tag}: {n} pairs, {len(keys)} proteins; NaN {int(res[cols].isna().sum().sum())}; corr(g, g_mc_mean) {np.corrcoef(res.g, res.g_mc_mean)[0, 1]:.4f}; corr(m, m_mc_mean) {np.corrcoef(res.m, res.m_mc_mean)[0, 1]:.4f}", flush=True)
    print("DONE %.1f min" % ((time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
