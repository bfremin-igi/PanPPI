"""E3 = E1 + TWO PROJECTIONS off one encoder + baked-in SNGP uncertainty (2026-11-12). Seed 141, as E1.

THE IDEA (user): the trained MaxSim converges onto Global and its real contribution is what it does to the
SHARED ENCODER, so treat it as a TRAINING DEVICE and never score it. "global + learnable maxsim basically
becomes the new global." Measured support:
  E. coli Global AUPR  LoGoPPI 0.5829 | run D, no rank loss 0.7172 ep4 -> 0.5487 ep11 COLLAPSING |
                       E1, rank loss 0.7070-0.7164, no collapse
  and the trained MaxSim is not worth reporting on its own: human Global-MaxSim gap 0.002 (LoGoPPI 0.175),
  and on DB5.5 it is ANTI-localised +0.71..+2.00 A vs exposure-matched random (CIs exclude 0) where LoGoPPI's
  untrained MaxSim is localised -0.85 A [-1.88, -0.18].
WIRING: proj_g is trained by the GLOBAL BCE ONLY and feeds Global (GP head) and maxsim_frozen -- LoGoPPI's
exact construction. proj_m is trained by the RANK LOSS, whose gradient reaches the ENCODER on purpose.
Model.forward returns `per` from proj_g (so the BCE trains proj_g) and, with return_proj, the proj_m residues
(so the rank loss trains proj_m and the encoder, never proj_g). threebranch_e3 asserts that isolation in 18
CPU checks, because a miswire fails SILENTLY -- it would train fine and simply be E1 with an unused Linear.
REPORTED BLEND IS UNCHANGED: 0.5 global + 0.5 maxsim. The frozen branch is deliberately named "maxsim" so the
existing calibration, blending and scoring paths apply with NO new weights to fit -- threebranch.py:25 records
why fitting them is harmful (MaxSim is worst in-distribution and best on transfer, so in-distribution fitting
downweights precisely the branch that generalises). maxsim_trained never reaches scoring.

ORIGINAL E2 HEADER:
E2 = E1 PLUS BAKED-IN UNCERTAINTY at two levels (2026-11-12). Seed 141, SAME as E1, deliberately: the only
difference from E1 is the architecture, so any change is attributable. (A different seed would confound
architecture with seed and would NOT measure training variance either, since E2 is not an E1 replicate.)
  (d) spectral norm on the shared projection (--sn-bound c, default 1.0 = TUnA's setting) + an RFF-GP in place
      of the Global branch's final Linear. Gives a calibrated probability WITH a distance-aware variance.
      This is the ONLY part that cannot be added post hoc: SNGP's guarantee comes from the representation
      being approximately bi-Lipschitz, a property of the LEARNED WEIGHTS, so it must be imposed in training.
  (e) a residue-level RFF-GP that qualifies MaxSim's per-residue importance scores (best[i] = max_j <x_i,y_j>)
      with an epistemic variance, a partner margin (top1-top2) and a selection gap. No residue labels are
      needed: the residue GP uses the unsupervised 'leverage' likelihood.
UNCERTAINTY IS FITTED POST-TRAINING, ON RANK 0, FROM THE EXISTING PER-PROTEIN CACHE. That is not a shortcut:
  * the precision matrix assumes a SETTLED feature extractor, which is false while the backbone is still
    moving (this is exactly why threebranch.py:43 deferred the GP head); after best.pt is loaded it is true;
  * precision/covariance are BUFFERS and DDP broadcasts buffers from rank 0 every forward, so accumulating
    inside DDP's forward would silently discard rank 1. Here the process group is already destroyed.
PRE-REGISTERED PREDICTIONS (written before any E2 result):
  Q1. ECE under domain shift improves vs E1 (the claim that is interpretable at n=1; AUPR differences below
      ~0.02 are NOT, our only noise prior being TUnA's seed SD 0.0077 yeast / 0.0102 ecoli).
  Q2. spectral norm reduces the E. coli Global collapse seen in run D (0.7172 ep4 -> 0.5487 ep11).
  Q3. residue var_res is higher on distant species than on human.
  Q4. SEPARATE, inference-only: LoGoPPI's UNTRAINED MaxSim ranks Propedia interface residues at least as well
      as E1's TRAINED MaxSim -- training the pooled mean need not sharpen per-residue attribution (XTR's
      finding for ColBERT). If Q4 holds, sparse pooling is the fix for residue attribution, not for AUPR.

ORIGINAL E1 HEADER:
E1 = "LoGoPPI-v2", first step (2026-11-10): LoGoPPI's two branches and recipe (copied from main_threebranch.py, run D)
PLUS (a) speed changes and (b) a ColBERT-style RANKING LOSS THROUGH MAXSIM. Run with --no-latent.
  (a) speed: --no-grad-ckpt (identical maths; memory is plentiful), --length-grouped (batches of similar-length pairs,
      less padding; a deviation from LoGoPPI's random sampler), larger per-GPU batch at the same effective 96.
  (b) --rank-loss-weight w: for every POSITIVE pair (a_i, b_i) in a GPU batch, softmax cross-entropy of
      maxsim(a_i, b_j) over all b_j in the batch (and symmetrically a_j vs b_i): the true partner must outscore the
      other in-batch proteins. In-batch candidates that are known training positives (either order) are masked.
      Raw maxsim scores, temperature --rank-temp (default 1, as ColBERT uses raw scores). This TRAINS through MaxSim,
      which LoGoPPI never does; Global keeps its BCE loss. Calibration and scoring unchanged (LoGoPPI's).
  (c) --select-on blend (ADDED + PRE-REGISTERED 2026-11-11, before E1 started; PLAN.md): epoch selection and early
      stopping on the calibrated equal-weight blend of the branches present instead of Global alone. LoGoPPI selects on
      Global because Global is its only trained branch; once MaxSim is trained (b) -- and once Latent exists (F) -- the
      Global-only rule cannot see the branch under test and can stop while it is still improving. --select-on global
      restores LoGoPPI's exact rule; both keys are logged every epoch and every epoch checkpoint is kept, so either
      rule can be scored afterwards.
Everything below is run D's script; run D itself keeps using main_threebranch.py unchanged.

ORIGINAL RUN D HEADER:
Run D (rebuilt 2026-11-10): LoGoPPI's recipe EXACTLY, plus our latent branch as a third, trained branch.

WHAT MATCHES LOGOPPI (verified from ext/LoGoPPI code + configs/training/cross_species.yaml, not the paper):
  data        their human_train_clean.csv (421,313), human_validation_selection.csv (checkpoint selection),
              human_validation_calibration.csv (post-hoc calibration), proteins.fasta, data/test/* species
  encoder     ESM-2 650M, every layer trainable except params unused for scoring (contact head, pooler,
              rotary-unused position table) -- esm2_global.py _freeze_unused_encoder_parameters
  residues    first 800 residues (max_residues 800), BOS/EOS excluded (exclude_special_tokens)
  head        Linear(1280->512) + dropout 0.1; Global = LN(3*512) -> Linear 512 -> ReLU -> Dropout 0.1 -> Linear 1
              on [a, b, |a-b|], mean of both orderings; HF normal(0, 0.02) init
  maxsim      parameter-free symmetric_maxsim on projected residues, NO loss, never trained
  loss        BCE with pos_weight 10 on the Global logit (+ the latent logit, our addition)
  optimiser   AdamW lr 2e-5, wd 0.01 (none on biases/norms), clip 1.0, linear warmup 10% then linear decay
              (get_linear_schedule_with_warmup), bf16 autocast, gradient checkpointing, effective batch 96
  selection   per epoch on the selection split, key (-Global AUPR, unweighted NLL, epoch) -> best.pt;
              early stopping after 3 epochs without AUPR improvement (training.py:477-555)
  calibration fit_calibration (training.py:1087): Global temperature+bias; MaxSim standardised with TRAIN-pair
              statistics then scale+bias; both by L-BFGS-B NLL on the calibration split; blend 0.5/0.5
OUR ADDITION: the latent branch (threebranch.LatentBranch) gets its own BCE, so it shapes the encoder too, and is
calibrated like Global (temperature + bias). Reported blends: LoGoPPI's (0.5 g + 0.5 m) and ours (g, m, l at 1/3).
NOT matched: their seed 42 / 3 GPUs (we use seed 141 / 2 GPUs, same effective batch).
CAVEAT carried from the old design: the global+maxsim number from this run is a SCORING-time ablation; the encoder
was also shaped by the latent loss, so a true LoGoPPI reproduction is the --no-latent run.

    torchrun --nproc_per_node 2 main_threebranch.py --esm-dir ext/esm2_650M_hf --data-dir <bundle>/data --run-dir R
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
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from threebranch_e3 import E3Head  # noqa: E402

FINAL_SPECIES = "yeast,ecoli,mouse,fly,worm,208964,224308,83332"   # 208964 P. aeruginosa, 224308 B. subtilis,
                                                                    # 83332 M. tuberculosis (in LoGoPPI's bundle)


def ddp_setup():
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if local >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK {local} but only {torch.cuda.device_count()} visible GPU(s): "
                               "DDP needs one GPU per rank")
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl")
        return True, int(os.environ["RANK"]), world, local
    return False, 0, 1, 0


def auroc(y, s):
    r = pd.Series(s).rank(method="average").values
    p, n = y.sum(), (~y).sum()
    return float("nan") if p == 0 or n == 0 else float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def aupr(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / y.sum()) if y.sum() else float("nan")


def nll(y, z):
    """Unweighted mean BCE of logits z, numerically stable (as logobert/scoring.py nll)."""
    z = np.asarray(z, float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def read_fasta(path):
    seqs, k, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if k is not None:
                    seqs[k] = "".join(buf)
                k, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if k is not None:
        seqs[k] = "".join(buf)
    return seqs


def read_pairs(path):
    d = pd.read_csv(path)
    d = d.rename(columns={"query": "a", "text": "b"})
    assert {"a", "b", "label"} <= set(d.columns), f"{path}: need query,text,label"
    return d[["a", "b", "label"]]


def sample(df, n, seed):
    """RANDOM sample, never .head(): LoGoPPI's files are SORTED BY LABEL."""
    return df if n is None or n >= len(df) else df.sample(n, random_state=seed).reset_index(drop=True)


class Model(torch.nn.Module):
    def __init__(self, esm_dir, d_enc, max_len, grad_ckpt=True, use_latent=True, encoder=None, tok=None,
                 use_sn=True, use_gp=True, sn_bound=1.0, rffs=1024, res_rffs=256, seed=123,
                 use_res_gp=False):
        super().__init__()
        if encoder is None:
            from transformers import AutoTokenizer, EsmModel
            tok = AutoTokenizer.from_pretrained(esm_dir)
            encoder = EsmModel.from_pretrained(esm_dir, add_pooling_layer=False)
            if grad_ckpt:
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.tok, self.esm = tok, encoder
        self.frozen = self._freeze_unused()
        self.head = E3Head(d_enc, use_gp=use_gp, sn_on_projg=use_sn, sn_bound=sn_bound,
                           rffs=rffs, res_rffs=res_rffs, use_res_gp=use_res_gp, seed=seed)
        self.max_len = max_len

    def _freeze_unused(self):
        """Parameters that never reach the loss: LoGoPPI freezes them, and DDP with find_unused_parameters=False
        would otherwise error on them. Rotary ESM never uses a position table."""
        rotary = getattr(getattr(self.esm, "config", None), "position_embedding_type", None) == "rotary"
        frozen = []
        for nm, q in self.esm.named_parameters():
            if nm.startswith(("contact_head.", "pooler.")) or (rotary and nm == "embeddings.position_embeddings.weight"):
                q.requires_grad_(False)
                frozen.append(nm)
        return frozen

    def encode(self, seqs, device):
        """[B, Lmax, d_enc] with BOS/EOS stripped, plus a residue mask. First max_len residues, dynamic padding."""
        seqs = [s[: self.max_len] for s in seqs]
        enc = self.tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        h = self.esm(**enc).last_hidden_state
        lens = torch.tensor([len(s) for s in seqs], device=device)
        lm = int(lens.max())
        h = h[:, 1:1 + lm, :]
        mask = torch.arange(lm, device=device)[None, :] < lens[:, None]
        return h, mask

    def forward(self, sa, sb, device, with_maxsim=False, return_proj=False):
        ha, ma = self.encode(sa, device)
        hb, mb = self.encode(sb, device)
        # Branch scores come from proj_g, so the BCE trains proj_g and NEVER proj_m.
        xa, xb = self.head.project_g(ha), self.head.project_g(hb)
        per = self.head.score_projected(xa, ma, xb, mb, with_maxsim=with_maxsim)
        if not return_proj:
            return per
        # The rank loss consumes what is returned here, so it MUST be the proj_m residues: that is what
        # keeps maxsim_frozen (on proj_g) free of any MaxSim gradient while the encoder still gets one.
        xma, xmb = self.head.rank_inputs(ha, hb)
        return per, (xma, ma, xmb, mb)


def lr_lambda(warm, total):
    """transformers.get_linear_schedule_with_warmup."""
    def f(step):
        if step < warm:
            return step / max(1, warm)
        return max(0.0, (total - step) / max(1, total - warm))
    return f


@torch.no_grad()
def fit_uncertainty(core, pairs, cache, dev, batch=64, holdout=0.05, seed=141):
    """Fit BOTH GPs on rank 0 from the per-protein cache, AFTER best.pt is loaded (settled features) and
    AFTER the process group is destroyed (so no DDP buffer broadcast can clobber the precision).

    The amplitude calibration is NOT optional: at the library default amplitude=1.0 the OOD/in-distribution
    variance ratio measured only ~2.2x on the synthetic probe, i.e. the uncertainty is effectively absent,
    and the dependence on amplitude is not monotonic so it cannot be guessed. See
    RFFGP.calibrate_amplitude and tests/test_e2_uncertainty.py.

    A 5% protein holdout is kept OUT of the precision accumulation as a NEGATIVE CONTROL: it is still
    in-distribution, so its variance should match the fitted proteins'. If it is already far higher, the fit
    is broken and nothing downstream is trustworthy."""
    head = core.head
    head.reset_all_precisions()
    ids = sorted(cache.keys())
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(np.array(ids, dtype=object), max(1, int(len(ids) * holdout)), replace=False).tolist())
    fit_ids = [k for k in ids if k not in hold]
    hold_ids = [k for k in ids if k in hold]
    print(f"  uncertainty: fitting on {len(fit_ids):,} proteins, {len(hold_ids):,} held out as a control",
          flush=True)

    if head.res_gp is not None:
        step = max(1, len(fit_ids) // 300)
        samp = torch.cat([cache[k].float() for k in fit_ids[::step]])[:4096].to(dev)
        print(f"  residue GP amplitude (median heuristic) = {head.res_gp.calibrate_amplitude(samp, reduce=False):.5f}",
              flush=True)
        for i in range(0, len(fit_ids), batch):
            x = torch.cat([cache[k].float() for k in fit_ids[i:i + batch]]).to(dev)
            head.res_gp.accumulate(head.res_gp.rff(x))
        head.res_gp.fit_covariance(reduce=False)

        def res_var(keys):
            v = []
            for i in range(0, len(keys), batch):
                x = torch.cat([cache[k].float() for k in keys[i:i + batch]]).to(dev)
                v.append(head.res_gp(x, get_var=True)[1])
            return torch.cat(v)

        v_fit, v_hold = res_var(fit_ids[::step]), res_var(hold_ids)
        ratio = float(v_hold.mean() / v_fit.mean().clamp_min(1e-12))
        print(f"  residue var: fitted {v_fit.mean():.3e}  held-out control {v_hold.mean():.3e}  ratio {ratio:.2f}x "
              f"({'OK' if ratio < 5 else 'SUSPECT -- control should be ~1x'})", flush=True)
    else:
        print("  residue GP OFF (DB5.5: per-residue attribution loses to a composition lookup) -- "
              "pair-level uncertainty only", flush=True)
        v_fit = v_hold = torch.zeros(1)
        ratio = float("nan")

    if getattr(head, "use_gp", False):
        A, B = pairs.a.values, pairs.b.values
        feats = []
        for i in range(0, len(pairs), batch):
            pa = torch.stack([cache[k].float().mean(0) for k in A[i:i + batch]]).to(dev)
            pb = torch.stack([cache[k].float().mean(0) for k in B[i:i + batch]]).to(dev)
            feats.append(head.g.feat(pa, pb))
        feats = torch.cat(feats)
        print(f"  pair GP amplitude = {head.g.gp.calibrate_amplitude(feats[:4096], reduce=False):.5f}", flush=True)
        for i in range(0, feats.shape[0], 4096):
            phi = head.g.gp.rff(feats[i:i + 4096])
            head.g.gp.accumulate(phi, (phi @ head.g.gp.out).squeeze(-1))
        head.g.gp.fit_covariance(reduce=False)
        print(f"  pair GP fitted on {feats.shape[0]:,} training pairs", flush=True)
    return {"res_var_fit": float(v_fit.mean()), "res_var_holdout": float(v_hold.mean()), "control_ratio": ratio}


@torch.no_grad()
def uncertainty_for(core, pairs, cache, dev, batch=64):
    """Per-pair uncertainty columns: the pair GP's variance and mean-field probability, plus a summary of the
    per-residue uncertainty over protein A's residues. Residue-level detail is written separately."""
    head = core.head
    out = {k: np.empty(len(pairs)) for k in ("global_var", "prob_mf", "res_var_mean", "res_var_max", "margin_mean")}
    A, B = pairs.a.values, pairs.b.values
    for i in range(0, len(pairs), batch):
        ea = [cache[k].float() for k in A[i:i + batch]]
        eb = [cache[k].float() for k in B[i:i + batch]]
        d = ea[0].shape[1]
        la, lb = [e.shape[0] for e in ea], [e.shape[0] for e in eb]
        xa = torch.zeros(len(ea), max(la), d, device=dev)
        xb = torch.zeros(len(eb), max(lb), d, device=dev)
        ma = torch.zeros(len(ea), max(la), dtype=torch.bool, device=dev)
        mb = torch.zeros(len(eb), max(lb), dtype=torch.bool, device=dev)
        for j, (q, r) in enumerate(zip(ea, eb)):
            xa[j, :q.shape[0]], ma[j, :q.shape[0]] = q.to(dev), True
            xb[j, :r.shape[0]], mb[j, :r.shape[0]] = r.to(dev), True
        sl = slice(i, i + len(ea))
        if getattr(head, "use_gp", False):
            o = head.score_projected(xa, ma, xb, mb, with_maxsim=False, get_var=True)
            out["global_var"][sl] = o["global_var"].float().cpu().numpy()
            out["prob_mf"][sl] = o["global_prob_mf"].float().cpu().numpy()
        else:
            out["global_var"][sl] = np.nan
            out["prob_mf"][sl] = np.nan
        rep = head.residue_report(xa, ma, xb, mb)
        n = ma.sum(1).clamp_min(1)
        out["margin_mean"][sl] = (rep["margin"].sum(1) / n).float().cpu().numpy()
        if "var_res" in rep:
            out["res_var_mean"][sl] = (rep["var_res"].sum(1) / n).float().cpu().numpy()
            out["res_var_max"][sl] = rep["var_res"].max(1).values.float().cpu().numpy()
        else:
            out["res_var_mean"][sl] = np.nan
            out["res_var_max"][sl] = np.nan
    return out


@torch.no_grad()
def embed_cache(model, ids, seqs, device, amp, batch=16):
    """Each protein encoded ONCE from its own sequence, then PROJECTED (the space MaxSim operates in, as
    LoGoPPI's residue cache). Stored fp16, as their cache_dtype."""
    model.eval()
    cache = {}
    ids = sorted(ids, key=lambda k: len(seqs[k]))      # length-sorted batches: far less padding
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        with amp():
            h, m = model.encode([seqs[k] for k in chunk], device)
            x = model.head.project(h)
        for j, k in enumerate(chunk):
            cache[k] = x[j, :int(m[j].sum())].half()
    return cache


@torch.no_grad()
def score_cached(model, pairs, cache, device, batch=64):
    """Raw per-branch outputs from the cache: only the head runs per pair, never the backbone."""
    model.eval()
    keys = ["global", "maxsim"] + (["latent"] if model.head.l is not None else [])
    out = {k: np.empty(len(pairs)) for k in keys}
    A, B = pairs.a.values, pairs.b.values
    for i in range(0, len(pairs), batch):
        ea, eb = [cache[k] for k in A[i:i + batch]], [cache[k] for k in B[i:i + batch]]
        d = ea[0].shape[1]
        la, lb = [e.shape[0] for e in ea], [e.shape[0] for e in eb]
        xa = torch.zeros(len(ea), max(la), d, device=device)
        xb = torch.zeros(len(eb), max(lb), d, device=device)
        ma = torch.zeros(len(ea), max(la), dtype=torch.bool, device=device)
        mb = torch.zeros(len(eb), max(lb), dtype=torch.bool, device=device)
        for j, (p, q) in enumerate(zip(ea, eb)):
            xa[j, :p.shape[0]], ma[j, :p.shape[0]] = p.float(), True
            xb[j, :q.shape[0]], mb[j, :q.shape[0]] = q.float(), True
        per = model.head.score_projected(xa, ma, xb, mb, with_maxsim=True)
        for k, v in per.items():
            out[k][i:i + batch] = v.float().cpu().numpy()
    return out


def fit_calibration(train_maxsim, cal, y):
    """training.py:1087 fit_calibration, plus the latent branch calibrated like Global."""
    from scipy.optimize import minimize
    mu, sd = float(np.mean(train_maxsim)), float(np.std(train_maxsim))
    assert sd > 0, "train MaxSim std must be positive"

    def fit(f):
        r = minimize(lambda p: nll(y, f(p)), np.zeros(2), method="L-BFGS-B", bounds=[(-10.0, 10.0), (None, None)])
        if not r.success:
            raise RuntimeError(f"calibration failed: {r.message}")
        return [float(v) for v in r.x]

    st = {"maxsim_train": {"mean": mu, "std": sd}}
    lt, b = fit(lambda p: cal["global"] / math.exp(p[0]) + p[1])
    st["global"] = {"temperature": math.exp(lt), "bias": b}
    zs = (cal["maxsim"] - mu) / sd
    ls, b = fit(lambda p: math.exp(p[0]) * zs + p[1])
    st["maxsim"] = {"scale": math.exp(ls), "bias": b}
    if "latent" in cal:
        lt, b = fit(lambda p: cal["latent"] / math.exp(p[0]) + p[1])
        st["latent"] = {"temperature": math.exp(lt), "bias": b}
    return st


def calibrate(raw, st):
    out = {"global": raw["global"] / st["global"]["temperature"] + st["global"]["bias"],
           "maxsim": st["maxsim"]["scale"] * (raw["maxsim"] - st["maxsim_train"]["mean"]) / st["maxsim_train"]["std"]
           + st["maxsim"]["bias"]}
    if "latent" in raw:
        out["latent"] = raw["latent"] / st["latent"]["temperature"] + st["latent"]["bias"]
    out["logoppi_g+m"] = 0.5 * (out["global"] + out["maxsim"])
    if "latent" in raw:
        out["three_g+m+l"] = (out["global"] + out["maxsim"] + out["latent"]) / 3.0
        out["g+l"] = 0.5 * (out["global"] + out["latent"])
        out["m+l"] = 0.5 * (out["maxsim"] + out["latent"])
    return out


def rank_loss(xa, ma, xb, mb, y, pos_mask, temp=1.0):
    """In-batch ColBERT-style loss through parameter-free MaxSim. S[i, j] = symmetric_maxsim(a_i, b_j) for all i, j in
    the batch; for each positive row i, cross-entropy of S[i, :] / temp at target j = i (and of S[:, i] at i = i).
    pos_mask[i, j] True where (a_i, b_j) is a known training positive with j != i: excluded as candidates."""
    from threebranch import symmetric_maxsim
    B = xa.shape[0]
    pos_rows = torch.nonzero(y > 0.5).flatten()
    if len(pos_rows) == 0:
        return xa.sum() * 0.0
    rows = []
    for i in range(B):        # row by row keeps memory at B x L x L instead of B x B x L x L
        rows.append(symmetric_maxsim(xa[i:i + 1].expand(B, -1, -1), ma[i:i + 1].expand(B, -1), xb, mb))
    S = torch.stack(rows, 0).float() / temp                   # [B, B]
    S = S.masked_fill(pos_mask, float("-inf"))
    tgt = pos_rows
    return 0.5 * (F.cross_entropy(S[pos_rows], tgt) + F.cross_entropy(S.T[pos_rows], tgt))


def length_grouped(order, lens_a, lens_b, batch, rng, mega=64):
    """Shuffle, cut into mega-chunks of `mega` batches, sort each chunk by the pair's longer protein, re-cut into
    batches, shuffle the batches. Same pairs per epoch, far less padding."""
    out = []
    step = batch * mega
    for i in range(0, len(order), step):
        ch = order[i:i + step]
        ch = ch[np.argsort(np.maximum(lens_a[ch], lens_b[ch]), kind="stable")]
        out += [ch[j:j + batch] for j in range(0, len(ch), batch)]
    perm = rng.permutation(len(out))
    return np.concatenate([out[k] for k in perm])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esm-dir", required=True)
    p.add_argument("--d-enc", type=int, default=1280, help="1280 for ESM-2 650M, 640 for 150M")
    p.add_argument("--data-dir", required=True, help="LoGoPPI bundle data/ (human_*.csv, proteins.fasta, test/)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16, help="per GPU (LoGoPPI: 16)")
    p.add_argument("--accum", type=int, default=3, help="16 x 3 x 2 GPUs = 96 (LoGoPPI: 16 x 2 x 3 GPUs)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-len", type=int, default=800)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--no-latent", action="store_true", help="two-branch run = a LoGoPPI reproduction")
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--subset", type=int, default=None, help="smoke: random train subset")
    p.add_argument("--eval-subset", type=int, default=None, help="smoke: random subset of every eval/test set")
    p.add_argument("--diag-species", default="yeast,ecoli", help="per-epoch transfer curve (display only)")
    p.add_argument("--final-species", default=FINAL_SPECIES)
    p.add_argument("--mem-probe", action="store_true",
                   help="one fwd/bwd on a worst-case batch (batch x 2 sequences at max_len), report peak, exit")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=141)
    p.add_argument("--rank-loss-weight", type=float, default=0.0)
    p.add_argument("--rank-temp", type=float, default=1.0)
    p.add_argument("--length-grouped", action="store_true")
    p.add_argument("--no-sn", action="store_true", help="disable spectral norm on the shared projection")
    p.add_argument("--no-gp", action="store_true", help="plain Linear Global head instead of the RFF-GP")
    p.add_argument("--sn-bound", type=float, default=1.0,
                   help="SNGP spectral norm bound c. 1.0 = TUnA's setting and the TIGHTEST; Liu et al. trade "
                        "expressiveness against distance awareness with this and sweep it on validation")
    p.add_argument("--res-gp", action="store_true",
                   help="also fit the RESIDUE-level GP. OFF by default on evidence: DB5.5 showed per-residue "
                        "MaxSim attribution loses to an amino-acid composition lookup (0.5447 vs 0.5833), so "
                        "a residue variance would be uncertainty about noise. Diagnostics only, not claims.")
    p.add_argument("--rffs", type=int, default=1024, help="random features for the PAIR GP")
    p.add_argument("--res-rffs", type=int, default=256,
                   help="random features for the RESIDUE GP. 256 not 1024: the synthetic probe separated OOD "
                        "~35x at 256 vs ~4.5x at 1024 (tests/test_e2_uncertainty.py). Re-validate on real data")
    p.add_argument("--select-on", choices=["global", "blend"], default="global",
                   help="epoch selection / early stop: 'global' = LoGoPPI's rule exactly; 'blend' = the calibrated "
                        "equal-weight blend of every branch present (PRE-REGISTERED 2026-11-11 for E1 and F)")
    a = p.parse_args()

    distributed, rank, world, local = ddp_setup()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    use_amp = a.precision == "bf16" and dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext
    run_dir = os.path.abspath(a.run_dir)
    os.makedirs(os.path.join(run_dir, "output", "checkpoints"), exist_ok=True)
    D = os.path.abspath(a.data_dir)

    tr = sample(read_pairs(os.path.join(D, "human_train_clean.csv")), a.subset, a.seed)
    sel = sample(read_pairs(os.path.join(D, "human_validation_selection.csv")), a.eval_subset, a.seed)
    cal = sample(read_pairs(os.path.join(D, "human_validation_calibration.csv")), a.eval_subset, a.seed)
    seqs = read_fasta(os.path.join(D, "proteins.fasta"))
    for nm, df in (("train", tr), ("selection", sel), ("calibration", cal)):
        miss = (set(df.a) | set(df.b)) - set(seqs)
        assert not miss, f"{nm}: {len(miss)} ids missing from proteins.fasta"
    if is_main:
        print(f"train {len(tr)} (pos {tr.label.mean():.4f})  selection {len(sel)}  calibration {len(cal)}  "
              f"proteins {len(seqs)}  >{a.max_len} aa: {np.mean([len(s) > a.max_len for s in seqs.values()]):.1%}",
              flush=True)

    model = Model(os.path.abspath(a.esm_dir), a.d_enc, a.max_len, grad_ckpt=not a.no_grad_ckpt,
                  use_latent=not a.no_latent, use_sn=not a.no_sn, use_gp=not a.no_gp,
                  sn_bound=a.sn_bound, rffs=a.rffs, res_rffs=a.res_rffs, seed=a.seed,
                  use_res_gp=a.res_gp).to(dev)
    nt = sum(q.numel() for q in model.parameters() if q.requires_grad)
    if is_main:
        print(f"world {world}  effective batch {a.batch * a.accum * world}  trainable {nt:,}  "
              f"latent {'off' if a.no_latent else 'on'}  max_len {a.max_len}  precision {a.precision}  "
              f"frozen (unused) {model.frozen}", flush=True)

    if a.mem_probe:
        L = a.max_len
        aa = "ACDEFGHIKLMNPQRSTVWY"
        rs = np.random.default_rng(0)
        fake = ["".join(rs.choice(list(aa), L)) for _ in range(a.batch)]
        torch.cuda.reset_peak_memory_stats(dev)
        model.train()
        t = time.time()
        with amp():
            per, (xa, ma_, xb, mb_) = model(fake, fake, dev, return_proj=True)
        loss = sum(F.binary_cross_entropy_with_logits(v.float(), torch.ones_like(v.float())) for v in per.values())
        if a.rank_loss_weight > 0:     # worst case: every row positive
            with amp():
                loss = loss + rank_loss(xa, ma_, xb, mb_, torch.ones(len(fake), device=dev),
                                        torch.zeros(len(fake), len(fake), dtype=torch.bool, device=dev))
        loss.backward()
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        tot = torch.cuda.get_device_properties(dev).total_memory / 2**30
        print(f"[rank {rank}] MEM PROBE batch {a.batch} x 2 x {L} residues: peak {peak:.1f} / {tot:.1f} GiB, "
              f"{time.time() - t:.1f} s fwd+bwd", flush=True)
        model.zero_grad(set_to_none=True)
        assert peak < 0.92 * tot, "worst-case batch leaves <8% headroom"
        if distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    if a.rank_loss_weight == 0:
        # CTRL (copy of main_e3.py, changed only here): with no rank loss proj_m never reaches the loss, and DDP
        # with find_unused_parameters=False would error on it. Freeze it; the optimizer filters on requires_grad.
        for q in model.head.proj_m.parameters():
            q.requires_grad_(False)
        assert not any(q.requires_grad for q in model.head.proj_m.parameters())
    core = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)
    decay, no_decay = [], []
    for nm, q in model.named_parameters():
        if q.requires_grad:
            (no_decay if (q.ndim == 1 or nm.endswith(".bias")) else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    per_rank = (len(tr) // (world * a.batch)) * a.batch         # equal backward passes on every rank (DDP)
    updates_per_epoch = (per_rank // a.batch) // a.accum
    total = updates_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(total * a.warmup_ratio), total))
    pw = torch.tensor([10.0], device=dev)

    sel_state = {"best_key": None, "best_aupr": None, "best_epoch": None, "no_improve": 0}
    start_epoch = 1
    resume_path = os.path.join(run_dir, "output", "checkpoints", "last.pt")
    if a.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        core.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        sel_state = st["sel_state"]
        start_epoch = int(st["epoch"]) + 1
        del st
        if is_main:
            print(f"RESUMING at epoch {start_epoch}; selection state {sel_state}", flush=True)

    log = None
    if is_main:
        with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump({**vars(a), "world_size": world, "updates_total": total}, f)
        log = open(os.path.join(run_dir, "output", "results.txt"), "a" if start_epoch > 1 else "w")
    rng = np.random.default_rng(a.seed)
    for _ in range(1, start_epoch):
        rng.permutation(len(tr))
    A, B, Y = tr.a.values, tr.b.values, tr.label.values
    LA = np.array([min(len(seqs[k]), a.max_len) for k in A])
    LB = np.array([min(len(seqs[k]), a.max_len) for k in B])
    rng_lg = np.random.default_rng(a.seed + 1000 * start_epoch + rank)
    pos_set = set(zip(A[Y == 1], B[Y == 1])) | set(zip(B[Y == 1], A[Y == 1]))
    t0 = time.time()
    stop = sel_state["no_improve"] >= a.patience

    for ep in range(start_epoch, a.epochs + 1):
        if stop:
            break
        model.train()
        ep_t0 = time.time()
        order = rng.permutation(len(tr))
        order = order[rank * per_rank:(rank + 1) * per_rank]
        if a.length_grouped:
            order = length_grouped(order, LA, LB, a.batch, rng_lg)
        tot_loss, steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, i in enumerate(range(0, per_rank, a.batch)):
            idx = order[i:i + a.batch]
            sync = (step + 1) % a.accum == 0
            ctx = model.no_sync() if (distributed and not sync) else contextlib.nullcontext()
            with ctx:
                y = torch.tensor(Y[idx], dtype=torch.float32, device=dev)
                with amp():
                    per, (xa, ma_, xb, mb_) = model([seqs[k] for k in A[idx]], [seqs[k] for k in B[idx]], dev,
                                                    return_proj=True)
                loss = sum(F.binary_cross_entropy_with_logits(v.float(), y, pos_weight=pw) for v in per.values())
                if a.rank_loss_weight > 0:
                    ai, bi = A[idx], B[idx]
                    pm = torch.tensor([[(j != i) and ((ai[i], bi[j]) in pos_set) for j in range(len(idx))]
                                       for i in range(len(idx))], device=dev)
                    with amp():
                        rl = rank_loss(xa, ma_, xb, mb_, y, pm, a.rank_temp)
                    loss = loss + a.rank_loss_weight * rl
                (loss / a.accum).backward()
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
            tot_loss += float(loss.detach()) * len(idx)
            if a.progress_every and (step + 1) % a.progress_every == 0 and is_main:
                done = (i + len(idx)) * world
                el = time.time() - ep_t0
                print(f"    ep{ep} {done}/{len(tr)} pairs  {done / el:.1f} pairs/s  epoch ETA "
                      f"{len(tr) / (done / el) / 3600:.2f} h  lr {sched.get_last_lr()[0]:.2e}  "
                      f"loss {tot_loss / (i + len(idx)):.4f}", flush=True)
        msg = f"epoch {ep}/{a.epochs} train {tot_loss / max(per_rank, 1):.4f} updates {steps} " \
              f"({(time.time() - t0) / 60:.1f} min)"

        if is_main:
            # SELECTION. --select-on global: exactly training.py:477, key (-Global AUPR, unweighted NLL, epoch).
            # --select-on blend (PRE-REGISTERED 2026-11-11, PLAN.md): the same key shape computed on the CALIBRATED
            # equal-weight blend of the branches present, because E1 TRAINS MaxSim (rank loss) and F adds Latent --
            # branches LoGoPPI's Global-only rule cannot see, so it can stop while they are still improving.
            # The per-epoch calibration is fitted on the calibration split and standardises MaxSim with
            # CALIBRATION-split statistics; final scoring keeps LoGoPPI's TRAIN statistics (scale and bias are refit
            # either way, so MaxSim's own ranking is unchanged; only the blend's mixing is affected).
            ids = set(sel.a) | set(sel.b)
            if a.select_on == "blend":
                ids |= set(cal.a) | set(cal.b)
            cache = embed_cache(core, ids, seqs, dev, amp)
            sc = score_cached(core, sel, cache, dev)
            sc_cal = score_cached(core, cal, cache, dev) if a.select_on == "blend" else None
            del cache
            ys = sel.label.values.astype(bool)
            g_aupr, g_nll = aupr(ys, sc["global"]), nll(ys, sc["global"])
            if a.select_on == "blend":
                st_ep = fit_calibration(sc_cal["maxsim"], sc_cal, cal.label.values.astype(float))
                c_ep = calibrate(sc, st_ep)
                bkey = "three_g+m+l" if "latent" in sc else "logoppi_g+m"
                k_aupr, k_nll = aupr(ys, c_ep[bkey]), nll(ys, c_ep[bkey])
            else:
                bkey, k_aupr, k_nll = "global", g_aupr, g_nll
            key = (-k_aupr, k_nll, ep)
            if sel_state["best_key"] is None or tuple(key) < tuple(sel_state["best_key"]):
                sel_state["best_key"], sel_state["best_epoch"] = list(key), ep
                torch.save(core.state_dict(), os.path.join(run_dir, "output", "checkpoints", "best.pt"))
            if sel_state["best_aupr"] is None or k_aupr > sel_state["best_aupr"]:
                sel_state["best_aupr"], sel_state["no_improve"] = k_aupr, 0
            else:
                sel_state["no_improve"] += 1
            stop = sel_state["no_improve"] >= a.patience
            msg += f" | SEL[{bkey}] {k_aupr:.4f} | global AUPR {g_aupr:.4f} maxsim(raw) {aupr(ys, sc['maxsim']):.4f}"
            if "latent" in sc:
                msg += f" latent {aupr(ys, sc['latent']):.4f}"
            msg += f" | best ep {sel_state['best_epoch']} no-improve {sel_state['no_improve']}"
            for sp in [s for s in a.diag_species.split(",") if s]:      # display only, never used for selection
                pr = sample(read_pairs(os.path.join(D, "test", f"{sp}_pairs.csv")), a.eval_subset, a.seed)
                sq = read_fasta(os.path.join(D, "test", f"{sp}.fasta"))
                cache = embed_cache(core, set(pr.a) | set(pr.b), sq, dev, amp)
                s2 = score_cached(core, pr, cache, dev)
                del cache
                yy = pr.label.values.astype(bool)
                msg += f" | {sp} " + " ".join(f"{k[0]} {aupr(yy, v):.4f}" for k, v in s2.items())
            torch.cuda.empty_cache()
            print("  " + msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            torch.save(core.state_dict(), os.path.join(run_dir, "output", "checkpoints", f"epoch_{ep:02d}.pt"))
            tmp = resume_path + ".tmp"
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "sel_state": sel_state, "epoch": ep}, tmp)
            os.replace(tmp, resume_path)
        if distributed:
            # all ranks receive rank 0's early-stop decision (training.py:551)
            flag = torch.tensor([int(stop)], device=dev)
            torch.distributed.broadcast(flag, 0)
            stop = bool(flag.item())
            torch.distributed.barrier()

    # Post-training (rank 0 only): calibration and final scoring of best.pt. Other ranks leave now so no
    # collective waits out an NCCL timeout during the long single-GPU stage.
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if not is_main:
        return
    if stop:
        print(f"early stop: {a.patience} epochs without selection-AUPR improvement", flush=True)
    core.load_state_dict(torch.load(os.path.join(run_dir, "output", "checkpoints", "best.pt"),
                                    map_location=dev, weights_only=False))
    print(f"FINAL: best epoch {sel_state['best_epoch']} (selection key --select-on {a.select_on}, AUPR "
          f"{-sel_state['best_key'][0]:.4f})", flush=True)
    human = set(tr.a) | set(tr.b) | set(sel.a) | set(sel.b) | set(cal.a) | set(cal.b)
    cache = embed_cache(core, human, seqs, dev, amp)
    unc_diag = fit_uncertainty(core, tr, cache, dev, seed=a.seed)
    raw_tr = score_cached(core, tr, cache, dev, batch=128)        # MaxSim TRAIN statistics (fit_calibration)
    raw_cal = score_cached(core, cal, cache, dev)
    raw_sel = score_cached(core, sel, cache, dev)
    # NOTE: the human cache is NOT freed here -- report("human_selection") below needs it for
    # uncertainty_for. Freeing it here was job 41247085's UnboundLocalError (smoke 2, 2026-11-12).
    st = fit_calibration(raw_tr["maxsim"], raw_cal, cal.label.values.astype(float))
    st.update({"blend": {"logoppi": "0.5 global + 0.5 maxsim", "three": "equal 1/3"},
               "best_epoch": sel_state["best_epoch"]})
    with open(os.path.join(run_dir, "calibration.json"), "w") as f:
        json.dump(st, f, indent=2)
    rows = []

    def report(tag, df, raw):
        y = df.label.values.astype(bool)
        c = calibrate(raw, st)
        print(f"  --- {tag} (n={len(y)}, prevalence {y.mean():.4f})", flush=True)
        for k, v in c.items():
            print(f"      {k:14} AUROC {auroc(y, v):.4f}  AUPR {aupr(y, v):.4f}", flush=True)
            rows.append(dict(set=tag, model=k, auroc=auroc(y, v), aupr=aupr(y, v), n=len(y), prev=float(y.mean())))
        u = uncertainty_for(core, df, _cache_for_report[0], dev)
        print(f"      UNCERTAINTY   pair var median {np.median(u['global_var']):.3e}  "
              f"residue var median {np.median(u['res_var_mean']):.3e}  "
              f"(Q3: distant species should exceed human_selection)", flush=True)
        rows.append(dict(set=tag, model="_uncertainty", auroc=np.nan, aupr=np.nan, n=len(y), prev=float(y.mean()),
                         pair_var=float(np.median(u["global_var"])),
                         res_var=float(np.median(u["res_var_mean"]))))
        pd.DataFrame({"a": df.a.values, "b": df.b.values, "label": df.label.values,
                      **{f"raw_{k}": v for k, v in raw.items()}, **{f"cal_{k}": v for k, v in c.items()},
                      **u}).to_csv(os.path.join(run_dir, f"scores_{tag}.tsv"), sep="\t", index=False)

    _cache_for_report = [cache]
    report("human_selection", sel, raw_sel)
    del cache
    _cache_for_report[0] = None
    torch.cuda.empty_cache()
    for sp in [s for s in a.final_species.split(",") if s]:
        pp = os.path.join(D, "test", f"{sp}_pairs.csv")
        if not os.path.exists(pp):
            print(f"  (no test file for {sp}, skipped)", flush=True)
            continue
        pr = sample(read_pairs(pp), a.eval_subset, a.seed)
        sq = read_fasta(os.path.join(D, "test", f"{sp}.fasta"))
        cache = embed_cache(core, set(pr.a) | set(pr.b), sq, dev, amp)
        _cache_for_report[0] = cache
        report(sp, pr, score_cached(core, pr, cache, dev))
        del cache
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    torch.save(core.state_dict(), os.path.join(run_dir, "output", "model"))
    print(f"\nwrote {run_dir}/summary.csv, calibration.json, scores_*.tsv  "
          f"({(time.time() - t0) / 3600:.2f} h total)", flush=True)
    if log is not None:
        log.close()


if __name__ == "__main__":
    main()
