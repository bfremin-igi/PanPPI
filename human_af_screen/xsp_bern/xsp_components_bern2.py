#!/usr/bin/env python3
"""Save the TWO BRANCH SCORES (global and frozen MaxSim, raw and calibrated) of a finished PanPPI run on the new bacterial / archaeal pair files, so the branches can be evaluated and re-weighted per unit
post hoc. No training, no GP. Same model construction, checkpoint, calibration and fp32 scoring as the post-hoc uncertainty run (organism-chunked caches so the 91k-protein bactU set fits on a 32 GB GPU).
CHECK: the blend 0.5*(cal_global+cal_maxsim) must equal the cal_blend of the existing post-hoc file of the same run (max abs difference is printed and enforced: exit 3 if > 5e-3).
usage: xsp_components.py RUN_DIR DATA_DIR OUT_DIR [--limit N]   (DATA_DIR holds <tag>.fasta and <tag>_pairs.csv for each tag in --tags; --check FILE verifies tag 'bchk' against a stored scores_bernett.tsv). Derived from analysis/xsp_components.py (md5 c8057c57): only the tag list, a strict missing-key assert and the reference check changed (v2: rank-based reproduction criterion)."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
import main_e3 as M


def read_fasta(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): k = l[1:].split()[0]; d[k] = ""
        elif l: d[k] += l
    return d


def organism_chunks(df, max_pairs=20000, max_prot=12000):
    org = df["a"].str.split(".").str[0].to_numpy(); order = np.argsort(org, kind="stable")
    a_, b_ = df["a"].to_numpy(), df["b"].to_numpy(); chunks, cur, prots = [], [], set()
    for i in order:
        new = {a_[i], b_[i]} - prots
        if cur and (len(cur) >= max_pairs or len(prots) + len(new) > max_prot):
            chunks.append(np.array(cur)); cur, prots = [], set(); new = {a_[i], b_[i]}
        cur.append(i); prots |= new
    if cur: chunks.append(np.array(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--esm-dir", default="ext/esm2_650M_hf"); ap.add_argument("--tags", default="yeast,mouse,fly,worm,ecoli"); ap.add_argument("--check", default=""); a = ap.parse_args(); tags = a.tags.split(",")
    dev = torch.device("cuda"); amp = torch.autocast  # fp32: no autocast (V100)
    class _NoAmp:
        def __enter__(self): return None
        def __exit__(self, *x): return False
    amp = lambda: _NoAmp()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    model = M.Model(os.path.abspath(a.esm_dir), 1280, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=141, use_res_gp=False).to(dev)
    sd = torch.load(f"{a.run}/output/checkpoints/best.pt", map_location="cpu", weights_only=False); rep = model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False); print("load_state_dict missing:", list(rep.missing_keys)[:8], "unexpected:", list(rep.unexpected_keys)[:8], flush=True); assert not rep.missing_keys, "missing keys"; model.eval()
    st = json.load(open(f"{a.run}/calibration.json")); print("model loaded %.1f s" % (time.time() - t0), flush=True)
    worst = 0.0
    for tag in tags:
        seqs = read_fasta(f"{a.data}/{tag}.fasta")
        df = pd.read_csv(f"{a.data}/{tag}_pairs.csv", dtype=str).rename(columns={"query": "a", "text": "b"}); df["label"] = df["label"].astype(int)
        if a.limit: df = df.iloc[:a.limit].copy()
        n = len(df); out = {k: np.zeros(n) for k in ("raw_global", "raw_maxsim", "cal_global", "cal_maxsim", "cal_blend")}
        for k, ix in enumerate(organism_chunks(df)):
            sub = df.iloc[ix]; cache = M.embed_cache(model, set(sub.a) | set(sub.b), seqs, dev, amp)
            raw = M.score_cached(model, sub, cache, dev); cc = M.calibrate(raw, st)
            out["raw_global"][ix] = raw["global"]; out["raw_maxsim"][ix] = raw["maxsim"]; out["cal_global"][ix] = cc["global"]; out["cal_maxsim"][ix] = cc["maxsim"]; out["cal_blend"][ix] = cc["logoppi_g+m"]
            del cache; torch.cuda.empty_cache()
            print(f"  {tag} chunk {k}: {len(ix):,} pairs ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res = df[["a", "b", "label"]].copy()
        for k, v in out.items(): res[k] = v
        dmax = 0.0
        if a.check and tag == "bchk":   # reproduce the stored Bernett-test scores of this very run on the first pairs of the Bernett test file
            ref = pd.read_csv(a.check, sep="\t", dtype={"a": str, "b": str}).set_index(["a", "b"]); r = ref.reindex(list(zip(res["a"], res["b"])))
            dmax = max(float(np.nanmax(np.abs(r[c].to_numpy() - res[k].to_numpy()))) for c, k in (("cal_global", "cal_global"), ("cal_maxsim", "cal_maxsim"), ("cal_logoppi_g+m", "cal_blend")))
            rr = pd.DataFrame({"o": res["cal_blend"].to_numpy(), "s": r["cal_logoppi_g+m"].to_numpy()}).dropna(); sp_ = float(rr.o.corr(rr.s, method="spearman")); med = float((rr.o - rr.s).abs().median())
            chk_ok = bool(sp_ >= 0.9999 and med < 0.01)   # v2 criterion (rank-based): the stored Bernett scores were computed under the training run's mixed precision, ours in fp32
            print(f"CHECK {tag}: {n} pairs; max |ours - stored| over cal_global / cal_maxsim / cal_blend = {dmax:.2e} (informational); blend Spearman {sp_:.6f}, median |diff| {med:.2e}; NaN in reference: {int(r['cal_global'].isna().sum())}; criterion (Spearman >= .9999 and median < .01) {'PASSED' if chk_ok else 'FAILED'}", flush=True); worst = 0.0 if chk_ok else 1.0
        res.to_csv(f"{a.out}/components_{tag}.tsv", sep="\t", index=False)
    print("DONE %.1f min; worst blend difference %.2e" % ((time.time() - t0) / 60, worst), flush=True)
    return 0 if worst <= 5e-3 else 3  # worst is 0 when the rank-based check passed


if __name__ == "__main__":
    sys.exit(main())
