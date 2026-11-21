"""POST-HOC UNCERTAINTY for a finished plain-head model (E3a, or CTRL): fit an RFF-GP on the trained head's penultimate features
AFTER training, then measure whether its variance is INFORMATIVE. (user, 2026-11-12: "no uncertainty during training, then add it
post-hoc after the training".)

WHY THIS IS A FAITHFUL ADD-ON, AND WHY IT IS NOT YET EVIDENCE OF GOOD UNCERTAINTY
  - The plain GlobalBranch is  logit = mlp[3]( 0.5*(h(a,b)+h(b,a)) ),  h = mlp[:3] o LayerNorm o cat[u, v, |u-v|]. The symmetrisation
    commutes with the final Linear, so the penultimate vector z = 0.5*(h(a,b)+h(b,a)) is recoverable and the model's own logit is EXACTLY
    mlp[3](z) (asserted per run, and in tests/test_e3a_nogp.py). A GP on z therefore sits on the quantity the model scored.
  - The GP here is the 'leverage' RFF-GP (no trained output weights): variance(z) = ridge * phi(z)^T (ridge I + Phi^T Phi)^-1 phi(z), with
    Phi the random-Fourier features of the TRAINING pairs' z and the kernel amplitude set by the median heuristic (calibrated, never
    guessed: threebranch_e3b.py documents what amplitude 1.0 did to E3).
  - NOT CLAIMED: that the variance is informative. SNGP's distance awareness rests on the spectral norm that E3a omits, and z is
    unconstrained, so whether this variance separates near from far has to be MEASURED. Readouts below decide it.

PRE-REGISTERED READOUTS (fixed before any number exists; the same set for every model scored)
  U0  identity check: |mlp[3](z) - the model's own Global logit| < 1e-3 on a sample of every set. If it fails, nothing below counts.
  U1  negative control: median variance on human_selection (never used to fit) vs a held-in sample of training pairs. A ratio far
      above ~3 means the fit is broken.
  U2  shift awareness: the species' variance should exceed human_selection's. Reported as the AUROC of variance for separating a
      species' pairs from human_selection pairs (0.5 = blind) and as the median ratio. Distant species (bacteria) should be higher.
  U3  error detection: AUROC of variance for flagging the model's mistakes at ONE operating point (the F1-optimal threshold on the
      CALIBRATION split, fixed before scoring any species), per set. 0.5 = useless.
  U4  selective prediction: F1 on the retained pairs after rejecting the highest-variance 10 / 25 / 50 %, against random rejection
      (mean of 20 draws). Informative only if it beats random.
  U5  calibration: ECE (15 equal-mass bins) of the calibrated blend probability, and of the mean-field-corrected probability
      sigmoid(logit / sqrt(1 + s*var)) with ONE scalar s fitted by NLL on the calibration split. If s ~ 0 or the NLL gain is ~ 0, the
      variance carries no calibration information.
  Single seed, single checkpoint: intervals are bootstrap over pairs and cover TEST-SET noise only, not training noise.

Usage (GPU; reads the run's own calibration.json and best.pt, writes only under --out-dir):
  python -I analysis/posthoc_uncertainty.py --run-dir runs_ft/e3a_nounc_650M_800_s141 --esm-dir ext/esm2_650M_hf \
      --data-dir data/logoppi/data
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import main_e3 as M  # noqa: E402  (Model, embed_cache, score_cached, calibrate, aupr, auroc, nll, read_*, sample)
from threebranch_e2 import RFFGP  # noqa: E402


# --------------------------------------------------------------------------------------------- features and the GP
def penultimate(gb, pa, pb):
    """z = 0.5*(h(a,b)+h(b,a)), h = mlp[:3] o LayerNorm o cat[u, v, |u-v|]. gb.mlp = [Linear, ReLU, Dropout, Linear]."""
    one = lambda u, v: gb.mlp[:3](gb.ln(torch.cat([u, v, (u - v).abs()], dim=-1)))
    return 0.5 * (one(pa, pb) + one(pb, pa))


@torch.no_grad()
def pair_features(head, cache, pairs, dev, batch=256):
    """z for every pair (cpu float32), from the per-protein cache, mean-pooled as the model pools."""
    head.eval()
    A, B = pairs.a.values, pairs.b.values
    out = []
    for i in range(0, len(pairs), batch):
        # .to(dev) PER ELEMENT: the cache can mix devices (M.embed_cache keeps its fp16 residues on the GPU, embed_pooled moves its
        # vectors to the CPU, and tr_cache = {**pooled, **cache}); stacking first crashed the first real GPU run (2026-11-17).
        pa = torch.stack([cache[k].float().mean(0).to(dev) for k in A[i:i + batch]])
        pb = torch.stack([cache[k].float().mean(0).to(dev) for k in B[i:i + batch]])
        out.append(penultimate(head.g, pa, pb).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_pooled(model, ids, seqs, dev, amp, batch=16):
    """Pooled projected vector per protein ONLY (1, d), computed exactly as pair_features would from the fp16 residue cache
    (round residues to fp16, then mean in fp32). The GP fit needs nothing else for the ~56k training proteins, and the full
    residue cache for them would be ~23 GB of RAM."""
    model.eval()
    out = {}
    ids = sorted(ids, key=lambda k: len(seqs[k]))
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        with amp():
            h, m = model.encode([seqs[k] for k in chunk], dev)
            x = model.head.project(h)
        for j, k in enumerate(chunk):
            out[k] = x[j, :int(m[j].sum())].half().float().mean(0, keepdim=True).cpu()
    return out


@torch.no_grad()
def fit_leverage_gp(Z, dev, rffs=1024, seed=0, amp_sample=4096, chunk=8192):
    """Leverage RFF-GP on the training features Z (N, d). Amplitude by the median heuristic on a random sample of Z."""
    gp = RFFGP(Z.shape[1], rffs=rffs, likelihood="leverage", seed=seed).to(dev)
    g = torch.Generator().manual_seed(seed)
    sub = Z[torch.randperm(Z.shape[0], generator=g)[:amp_sample]].to(dev)
    amp = gp.calibrate_amplitude(sub, reduce=False)
    for i in range(0, Z.shape[0], chunk):
        gp.accumulate(gp.rff(Z[i:i + chunk].to(dev)))
    gp.fit_covariance(reduce=False)
    return gp, amp


@torch.no_grad()
def gp_variance(gp, Z, dev, chunk=8192):
    return torch.cat([gp(Z[i:i + chunk].to(dev), get_var=True)[1].float().cpu() for i in range(0, Z.shape[0], chunk)]).numpy()


# --------------------------------------------------------------------------------------------- metrics
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def ece(y, p, bins=15):
    """Expected calibration error with equal-MASS bins (equal-width bins are nearly empty at prevalence ~0.1)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    tot = 0.0
    for idx in np.array_split(o, bins):
        if len(idx):
            tot += abs(y[idx].mean() - p[idx].mean()) * len(idx)
    return float(tot / len(y))


def fit_mf_scale(logit, var, y):
    """ONE scalar s >= 0 for sigmoid(logit / sqrt(1 + s*var)), by NLL. Returns (s, nll_at_s, nll_at_0)."""
    from scipy.optimize import minimize_scalar
    logit, var, y = np.asarray(logit, float), np.asarray(var, float), np.asarray(y, float)
    f = lambda ls: M.nll(y, logit / np.sqrt(1.0 + math.exp(ls) * var))
    r = minimize_scalar(f, bounds=(-14.0, 14.0), method="bounded")
    s = math.exp(r.x)
    return float(s), float(f(r.x)), float(M.nll(y, logit))


def f1(y, pred):
    tp = float(np.sum(pred & y))
    p, r = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def best_threshold(y, p):
    """F1-optimal probability threshold, searched on quantiles (used on the CALIBRATION split only)."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 120)))
    return float(max(grid, key=lambda t: f1(y, p >= t)))


def error_auroc(y, p, var, thr):
    """AUROC of the variance for flagging the model's mistakes at threshold thr. NaN if there is only one class."""
    err = ((p >= thr) != y.astype(bool))
    return M.auroc(err, var)


def f1_at_coverage(y, p, var, thr, cov, rng, n_rand=20):
    """F1 on the retained pairs after rejecting the highest-variance (1-cov) fraction, vs random rejection (mean of n_rand)."""
    n = len(y)
    k = max(1, int(round(cov * n)))
    keep = np.argsort(var, kind="stable")[:k]
    yb = y.astype(bool)
    by_var = f1(yb[keep], p[keep] >= thr)
    rnd = float(np.mean([f1(yb[ix], p[ix] >= thr) for ix in (rng.permutation(n)[:k] for _ in range(n_rand))]))
    return by_var, rnd


def boot_ci(fn, n, rng, B=500):
    vals = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        v = fn(ix)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="a finished plain-head run (has calibration.json and output/checkpoints/best.pt)")
    ap.add_argument("--esm-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="default <run-dir>/output/checkpoints/best.pt")
    ap.add_argument("--out-dir", default=None, help="default <run-dir>/posthoc_gp")
    ap.add_argument("--species", default=M.FINAL_SPECIES)
    ap.add_argument("--eval-subset", type=int, default=None)
    ap.add_argument("--rffs", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="fp32 on Turing/Pascal cards (no bf16)")
    ap.add_argument("--seed", type=int, default=141)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (a.dtype == "bf16" and dev.type == "cuda") \
        else contextlib.nullcontext
    run = os.path.abspath(a.run_dir)
    ck = a.ckpt or os.path.join(run, "output", "checkpoints", "best.pt")
    out = os.path.abspath(a.out_dir or os.path.join(run, "posthoc_gp"))
    os.makedirs(out, exist_ok=True)
    st = json.load(open(os.path.join(run, "calibration.json")))
    rng = np.random.default_rng(a.seed)
    D = os.path.abspath(a.data_dir)
    t0 = time.time()

    tr = M.read_pairs(os.path.join(D, "human_train_clean.csv"))
    sel = M.sample(M.read_pairs(os.path.join(D, "human_validation_selection.csv")), a.eval_subset, a.seed)
    cal = M.sample(M.read_pairs(os.path.join(D, "human_validation_calibration.csv")), a.eval_subset, a.seed)
    if a.eval_subset:
        tr = M.sample(tr, 20 * a.eval_subset, a.seed)
    seqs = M.read_fasta(os.path.join(D, "proteins.fasta"))

    # The plain head: exactly the E3a / CTRL flags (--no-latent --no-sn --no-gp).
    model = M.Model(os.path.abspath(a.esm_dir), 1280, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False,
                    seed=a.seed, use_res_gp=False).to(dev)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    print(f"loaded {ck}; run {run}", flush=True)

    eval_ids = set(sel.a) | set(sel.b) | set(cal.a) | set(cal.b)
    cache = M.embed_cache(model, eval_ids, seqs, dev, amp)            # full residues: MaxSim needs them
    pooled = embed_pooled(model, (set(tr.a) | set(tr.b)) - eval_ids, seqs, dev, amp)   # train proteins: pooled vector only
    tr_cache = {**pooled, **cache}
    print(f"caches: {len(cache):,} proteins with residues, {len(pooled):,} pooled-only ({(time.time() - t0) / 60:.1f} min)",
          flush=True)

    Ztr = pair_features(model.head, tr_cache, tr, dev)
    gp, amplitude = fit_leverage_gp(Ztr, dev, rffs=a.rffs, seed=a.seed)
    print(f"post-hoc GP: {Ztr.shape[0]:,} training pairs, z dim {Ztr.shape[1]}, median-heuristic amplitude {amplitude:.4f}", flush=True)

    def score(df, c):
        """raw branches, z, variance, calibrated blend logit, and the U0 identity check."""
        raw = M.score_cached(model, df, c, dev)
        Z = pair_features(model.head, c, df, dev)
        with torch.no_grad():
            ident = float(np.abs(model.head.g.mlp[3](Z.to(dev)).squeeze(-1).float().cpu().numpy() - raw["global"]).max())
        cc = M.calibrate(raw, st)
        return raw, Z, gp_variance(gp, Z, dev), cc["logoppi_g+m"], ident

    # U1 negative control, and the fitting of the single mean-field scalar and the F1 threshold, all on CALIBRATION
    _, Zc, vc, lc, id_c = score(cal, cache)
    yc = cal.label.values.astype(bool)
    pc = sigmoid(lc)
    thr = best_threshold(yc, pc)
    s_mf, nll_s, nll_0 = fit_mf_scale(lc, vc, yc.astype(float))
    held_in = tr.sample(min(2000, len(tr)), random_state=a.seed)
    v_in = gp_variance(gp, pair_features(model.head, tr_cache, held_in, dev), dev)
    del tr_cache, pooled
    summary = {"ckpt": ck, "n_train_gp": int(Ztr.shape[0]), "amplitude": amplitude, "f1_threshold": thr,
               "mf_scale_s": s_mf, "mf_nll_at_s": nll_s, "mf_nll_at_0": nll_0, "identity_max_abs_calibration": id_c}
    print(f"  calibration split: F1-optimal threshold {thr:.3f}; mean-field s {s_mf:.4g} (NLL {nll_0:.4f} -> {nll_s:.4f}); "
          f"identity |mlp[3](z) - logit| max {id_c:.2e}", flush=True)

    rows = []
    var_by = {}

    def report(tag, df, c):
        raw, Z, v, lg, ident = score(df, c)
        y = df.label.values.astype(bool)
        p = sigmoid(lg)
        p_mf = sigmoid(lg / np.sqrt(1.0 + s_mf * v))
        var_by[tag] = v
        e_auc = error_auroc(y, p, v, thr)
        ci = boot_ci(lambda ix: error_auroc(y[ix], p[ix], v[ix], thr), len(y), rng, B=300)
        row = dict(set=tag, n=len(y), prev=float(y.mean()), aupr_blend=M.aupr(y, lg), auroc_blend=M.auroc(y, lg),
                   aupr_mf=M.aupr(y, p_mf), ece=ece(y, p), ece_mf=ece(y, p_mf), nll=M.nll(y.astype(float), lg),
                   nll_mf=M.nll(y.astype(float), lg / np.sqrt(1.0 + s_mf * v)), var_median=float(np.median(v)),
                   err_auroc=e_auc, err_auroc_lo=ci[0], err_auroc_hi=ci[1], identity_max_abs=ident,
                   f1_full=f1(y, p >= thr))
        for cov in (0.9, 0.75, 0.5):
            fv, fr = f1_at_coverage(y, p, v, thr, cov, rng)
            row[f"f1_cov{int(cov * 100)}_var"], row[f"f1_cov{int(cov * 100)}_random"] = fv, fr
        rows.append(row)
        pd.DataFrame({"a": df.a.values, "b": df.b.values, "label": df.label.values, "cal_blend": lg, "var": v, "prob": p,
                      "prob_mf": p_mf}).to_csv(os.path.join(out, f"scores_{tag}_posthoc.tsv"), sep="\t", index=False)
        print(f"  {tag:16} n {len(y):6} prev {y.mean():.3f} | AUPR {row['aupr_blend']:.4f} | ECE {row['ece']:.4f} -> mf "
              f"{row['ece_mf']:.4f} | var median {row['var_median']:.3e} | err-AUROC {e_auc:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | "
              f"F1@cov50 var {row['f1_cov50_var']:.3f} vs random {row['f1_cov50_random']:.3f} | identity {ident:.1e}", flush=True)

    report("human_selection", sel, cache)
    summary["var_median_held_in_train"] = float(np.median(v_in))
    summary["control_ratio_sel_over_train"] = float(np.median(var_by["human_selection"]) / max(float(np.median(v_in)), 1e-30))
    print(f"  U1 negative control: median var human_selection / held-in train = {summary['control_ratio_sel_over_train']:.2f}x "
          f"({'OK' if summary['control_ratio_sel_over_train'] < 3 else 'SUSPECT'})", flush=True)
    del cache
    torch.cuda.empty_cache()

    for sp in [s for s in a.species.split(",") if s]:
        pp = os.path.join(D, "test", f"{sp}_pairs.csv")
        if not os.path.exists(pp):
            print(f"  (no test file for {sp}, skipped)", flush=True)
            continue
        pr = M.sample(M.read_pairs(pp), a.eval_subset, a.seed)
        sq = M.read_fasta(os.path.join(D, "test", f"{sp}.fasta"))
        c = M.embed_cache(model, set(pr.a) | set(pr.b), sq, dev, amp)
        report(sp, pr, c)
        del c
        torch.cuda.empty_cache()

    # U2 shift awareness
    vs = var_by["human_selection"]
    for r in rows:
        if r["set"] != "human_selection":
            v = var_by[r["set"]]
            lab = np.r_[np.zeros(len(vs), bool), np.ones(len(v), bool)]
            r["shift_auroc_vs_selection"] = M.auroc(lab, np.r_[vs, v])
            r["var_ratio_vs_selection"] = r["var_median"] / max(float(np.median(vs)), 1e-30)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(out, "posthoc_summary.csv"), index=False)
    json.dump(summary, open(os.path.join(out, "posthoc_gp.json"), "w"), indent=2)
    print("\n--- U2 shift awareness (AUROC of variance, species vs human_selection; ratio of medians)", flush=True)
    for r in rows:
        if "shift_auroc_vs_selection" in r:
            print(f"  {r['set']:10} shift-AUROC {r['shift_auroc_vs_selection']:.3f}  var ratio {r['var_ratio_vs_selection']:.2f}x  "
                  f"AUPR {r['aupr_blend']:.4f}", flush=True)
    print(f"\nwrote {out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
