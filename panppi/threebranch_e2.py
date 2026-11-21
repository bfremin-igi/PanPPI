"""E2 head: E1's architecture + BAKED-IN UNCERTAINTY at two levels.

  (1) BINARY / pair level  -- SNGP: spectral norm on the shared projection + a random-Fourier-feature GP in
      place of the Global branch's final Linear. Gives a calibrated probability WITH a variance, and the
      variance is distance aware (high far from the training manifold) rather than merely low-confidence.
  (2) INTERACTION-RESIDUE level -- the same distance awareness applied to each residue's projected embedding,
      so every residue that MaxSim nominates as important carries its own epistemic uncertainty.

WHY THIS CANNOT BE BOLTED ONTO E1 AFTER THE FACT. Post-hoc last-layer Laplace fits a confidence model on top
of whatever representation training happened to produce. SNGP's guarantee comes from the representation being
approximately bi-Lipschitz, which is a property of the LEARNED WEIGHTS and so has to be imposed DURING
training by spectral normalisation. Without it, a Laplace/GP variance stays over-confident exactly where we
need it most: distant species, peptides, GPCRs. Everything else in the uncertainty stack (MC dropout, seed
ensembles, argmax margins, post-hoc Laplace) genuinely can be added later, and should be -- this file is only
for the part that cannot.

HOW RESIDUE UNCERTAINTY IS DEFINED, given we have NO residue-level labels in the human training data.
MaxSim already yields a per-residue importance score: best[i] = max_j <x_i, y_j>, i.e. "my residue i's best
match in the partner". That is the residue selection we want to qualify. We attach three signals to it:
  var_res[i]   EPISTEMIC. RFF-GP leverage of residue i's own projected embedding against the training
               residue distribution (gaussian likelihood => prob_multiplier 1 => precision is the plain Gram
               sum, which needs no labels). Large = this residue looks unlike anything seen in training.
               ONLY MEANINGFUL BECAUSE OF THE SPECTRAL NORM: without it, embedding distance is not tied to
               input distance and the leverage is uninformative.
  margin[i]    WHICH PARTNER. best[i] is a max over the partner's residues; top1 - top2 says whether the
               identity of the binding partner residue is pinned down or a near-tie. Distinct question from
               "is residue i in the interface at all", and free.
  sel_gap[i]   WHICH RESIDUE. best[i] - (mean of best over own residues), i.e. how far this residue stands
               out from its own protein's background. Free.
UNVALIDATED PREMISE, STATED LOUDLY: nobody has yet shown that best[] ranks true interface residues above
chance. The latent branch failed that test (job 71051: AUROC 0.532 vs a 0.525 composition baseline). If
best[] is also at chance then these are uncertainties about noise. analysis/latent_attention.py + Propedia's
annotated interface residues answer this inference-only; RUN THAT FIRST.

DDP CORRECTNESS -- the thing most likely to break silently.
  * `precision` and `covariance` are BUFFERS. torch DDP broadcasts buffers from rank 0 on every forward
    (broadcast_buffers=True, the default), so a precision matrix accumulated inside DDP's forward would have
    rank 1's contribution silently overwritten every step. Therefore the precision pass runs OUTSIDE DDP, on
    the unwrapped module, and the shards are combined with ONE all_reduce(SUM). That is exact: precision is a
    sum of outer products, so summing per-rank partial sums is identical to a single-process pass.
  * momentum < 0 selects the single-pass exact sum (no moving average), which is what a post-training
    accumulation wants.
  * covariance is formed ONCE, after the all_reduce, by pinv. Calling it per-rank before the reduce would
    give each rank a different, wrong covariance.
  * known gotcha carried over from TUnA: load checkpoints AFTER building the trainer, and reset GP precision
    when resuming, or a requeued run accumulates precision twice.
"""
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils import spectral_norm

from .threebranch import GlobalBranch, hf_init, masked_mean, symmetric_maxsim


class RFFGP(nn.Module):
    """Vanilla random-Fourier-feature GP layer, semantics mirrored from
    ref/uncertaintyAwareDeepLearn/classic_rffs.py (VanillaRFFLayer) but self-contained, DDP-aware, and with
    `likelihood` extended to an unsupervised 'leverage' mode for the residue head.

    likelihood:
      'binary_logistic'  precision weighted by p(1-p)  -- the pair head (we have labels)
      'leverage'         precision weighted by 1       -- the residue head (no labels; plain Gram sum)
    """

    def __init__(self, in_features, rffs=1024, likelihood="binary_logistic",
                 ridge_penalty=1e-3, amplitude=1.0, seed=123):
        super().__init__()
        if rffs % 2 or rffs < 2:
            raise ValueError("rffs must be an even number > 1")
        if likelihood not in ("binary_logistic", "leverage"):
            raise ValueError(likelihood)
        self.rffs, self.likelihood = rffs, likelihood
        self.ridge_penalty, self.amplitude = ridge_penalty, amplitude
        self.num_freqs = rffs // 2
        self.feature_scale = math.sqrt(2.0 / self.num_freqs)
        g = torch.Generator().manual_seed(seed)          # identical frequencies on every rank
        self.register_buffer("weight_mat", torch.randn(in_features, self.num_freqs, generator=g))
        self.out = nn.Parameter(torch.zeros(rffs, 1)) if likelihood == "binary_logistic" else None
        if self.out is not None:
            nn.init.normal_(self.out, 0.0, 0.05)
        self.register_buffer("precision", torch.zeros(rffs, rffs))
        self.register_buffer("covariance", torch.zeros(rffs, rffs))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def rff(self, x):
        z = self.amplitude * (x.float() @ self.weight_mat.float())
        return self.feature_scale * torch.cat([torch.cos(z), torch.sin(z)], dim=-1)

    @torch.no_grad()
    def calibrate_amplitude(self, x, reduce=True):
        """Set the kernel amplitude (inverse lengthscale) by the MEDIAN HEURISTIC on a sample of REAL
        features, and return it.

        THIS IS NOT OPTIONAL. amplitude controls how fast the kernel decays with distance, so it decides
        whether the leverage variance can tell "far from training" from "near". Measured on a synthetic
        cluster (600 pts, 32-d, tests/test_e2_uncertainty.py), the out-of-distribution to in-distribution
        variance ratio was:
            amp 0.1 / 1024 RFFs ->   1.2x     amp 1.0 / 1024 ->   2.2x     amp 3.0 / 1024 -> 828.8x
            amp 0.1 /  256 RFFs ->   9.1x     amp 1.0 /  256 ->   2.4x     amp 0.3 /  256 ->  32.6x
        i.e. AT THE DEFAULT amplitude=1.0 THE SIGNAL IS ESSENTIALLY ABSENT, and the dependence is not even
        monotonic, so it cannot be guessed. A mis-set amplitude yields a null uncertainty result that looks
        like "the method does not work" rather than "the kernel was mis-scaled".

        Median heuristic: lengthscale = median pairwise distance, amplitude = 1 / lengthscale. Under DDP the
        median is taken on each rank's sample and averaged, which is close enough for a kernel scale (an
        exact global median would need a full gather for no practical gain)."""
        x = x.float().reshape(-1, x.shape[-1])
        if x.shape[0] > 2048:                      # subsample: the median is stable long before this
            x = x[torch.randperm(x.shape[0], device=x.device)[:2048]]
        d = torch.cdist(x, x)
        med = d[torch.triu(torch.ones_like(d), diagonal=1) > 0].median().clamp_min(1e-6)
        amp = (1.0 / med).item()
        if reduce and dist.is_available() and dist.is_initialized():
            t = torch.tensor([amp], device=x.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            amp = (t / dist.get_world_size()).item()
        self.amplitude = amp
        return amp

    def forward(self, x, update_precision=False, get_var=False):
        """x: [N, in_features]. Returns logits, or (logits, var) when get_var."""
        phi = self.rff(x)
        logits = (phi @ self.out).squeeze(-1) if self.out is not None else None
        if update_precision:
            self.accumulate(phi, logits)
        if get_var:
            if not bool(self.fitted):
                raise RuntimeError("call fit_covariance() before asking for a variance")
            with torch.no_grad():
                var = (phi * (self.ridge_penalty * (self.covariance.float() @ phi.T).T)).sum(-1)
            return logits, var
        return logits

    @torch.no_grad()
    def accumulate(self, phi, logits=None):
        """Add this shard's outer-product sum. Call OUTSIDE DDP's forward (see module docstring)."""
        if self.likelihood == "binary_logistic":
            p = torch.sigmoid(logits.float())
            phi = torch.sqrt((p * (1 - p)).clamp_min(1e-12)).unsqueeze(-1) * phi
        self.precision += phi.T @ phi

    @torch.no_grad()
    def reset_precision(self):
        self.precision.zero_()
        self.covariance.zero_()
        self.fitted.fill_(False)

    @torch.no_grad()
    def fit_covariance(self, reduce=True):
        """ONE all_reduce of the accumulated precision, then invert. Must be called on every rank."""
        if reduce and dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.precision, op=dist.ReduceOp.SUM)
        eye = torch.eye(self.rffs, device=self.precision.device, dtype=torch.float32)
        self.covariance.copy_(torch.linalg.pinv(self.ridge_penalty * eye + self.precision.float()))
        self.fitted.fill_(True)


def mean_field(logit, var):
    """SNGP mean-field correction (model.py:213 mean_field_average): shrink the logit toward 0 in proportion
    to its epistemic variance, then sigmoid."""
    return torch.sigmoid(logit / torch.sqrt(1.0 + (math.pi / 8.0) * var))


class GlobalGPBranch(nn.Module):
    """LoGoPPI's GlobalBranch with its final Linear replaced by an RFF-GP.

    The symmetrisation 0.5*(one(a,b) + one(b,a)) is kept, and is EXACT here for the same reason as in the
    linear case: the GP output layer is linear in the RFF features, so averaging the two orderings' logits
    equals applying the output layer to the averaged RFF features. The variance is taken on that averaged
    feature, so there is exactly one variance per pair (not two to reconcile)."""

    def __init__(self, d, hid=512, rffs=1024, seed=123):
        super().__init__()
        self.ln = nn.LayerNorm(3 * d)
        self.body = nn.Sequential(spectral_norm(nn.Linear(3 * d, hid)), nn.ReLU(), nn.Dropout(0.1))
        self.gp = RFFGP(hid, rffs=rffs, likelihood="binary_logistic", seed=seed)

    def feat(self, a, b):
        one = lambda u, v: self.body(self.ln(torch.cat([u, v, (u - v).abs()], dim=-1)))
        return 0.5 * (one(a, b) + one(b, a))

    def forward(self, pa, pb, update_precision=False, get_var=False):
        return self.gp(self.feat(pa, pb), update_precision=update_precision, get_var=get_var)


class ScaledSN(nn.Module):
    """spectral_norm pins sigma(W) to exactly 1, i.e. the SNGP bound c = 1 -- the TIGHTEST setting and the end
    of the range that risks underfitting. Liu et al. describe c as the knob trading expressiveness against
    distance awareness (too small collapses the block toward identity; too large loses bi-Lipschitz) and
    recommend sweeping c and the kernel amplitude on validation. This exposes c. c scales the layer's whole
    output, bias included, which is a harmless reparameterisation training absorbs.
    TUnA (tunaplus/model.py) uses plain spectral_norm, i.e. c = 1, in our own domain -- hence the default."""

    def __init__(self, lin, c=1.0):
        super().__init__()
        self.lin, self.c = spectral_norm(lin), float(c)

    @property
    def weight(self):
        return self.lin.weight

    def forward(self, x):
        return self.c * self.lin(x) if self.c != 1.0 else self.lin(x)


class E2Head(nn.Module):
    """E1's head with (a) a spectral-normed shared projection, (b) a GP Global branch, (c) a residue-level
    GP that qualifies MaxSim's per-residue importance scores. MaxSim itself stays parameter-free.

    use_sn=False / use_gp=False degrade this to E1's head exactly, which is what the ablation tests assert."""

    # res_rffs=256 NOT 512/1024: measured separation on the synthetic probe was ~35x at 256 RFFs vs ~4.5x at
    # 1024 (tests/test_e2_uncertainty.py). More random features let the training set span more directions,
    # which flattens the leverage contrast. MUST be re-validated on real projected residues.
    def __init__(self, d_enc, d_proj=512, rffs=1024, res_rffs=256, use_sn=True, use_gp=True,
                 sn_bound=1.0, seed=123):
        super().__init__()
        proj = nn.Linear(d_enc, d_proj)
        proj.apply(hf_init)
        self.proj = ScaledSN(proj, sn_bound) if use_sn else proj
        self.l = None          # compat: main_e1.score_cached tests `model.head.l is not None` for the latent branch
        self.drop = nn.Dropout(0.1)
        self.use_gp = use_gp
        if use_gp:
            self.g = GlobalGPBranch(d_proj, rffs=rffs, seed=seed)
            self.g.ln.apply(hf_init)
        else:
            self.g = GlobalBranch(d_proj)
            self.g.apply(hf_init)
        self.res_gp = RFFGP(d_proj, rffs=res_rffs, likelihood="leverage", seed=seed + 1)

    def project(self, h):
        return self.drop(self.proj(h))

    def score_projected(self, xa, ma, xb, mb, with_maxsim=True, update_precision=False, get_var=False):
        pa, pb = masked_mean(xa, ma), masked_mean(xb, mb)
        out = {}
        if self.use_gp and get_var:
            logit, var = self.g(pa, pb, update_precision=update_precision, get_var=True)
            out["global"], out["global_var"] = logit, var
            out["global_prob_mf"] = mean_field(logit, var)
        elif self.use_gp:
            out["global"] = self.g(pa, pb, update_precision=update_precision)
        else:
            out["global"] = self.g(pa, pb)
        if with_maxsim:
            out["maxsim"] = symmetric_maxsim(xa, ma, xb, mb)
        return out

    def forward(self, ha, ma, hb, mb, with_maxsim=True, update_precision=False, get_var=False):
        return self.score_projected(self.project(ha), ma, self.project(hb), mb,
                                    with_maxsim, update_precision, get_var)

    # ---- interaction-residue level -------------------------------------------------------------------
    def residue_report(self, xa, ma, xb, mb, get_var=True):
        """Per-residue importance AND its uncertainty, for protein A's residues. All tensors [B, La].
          best     MaxSim's per-residue importance (max over the partner's residues)
          sel_gap  best - mean(best over own valid residues): how far this residue stands out
          margin   top1 - top2 over the PARTNER's residues: is the binding partner pinned down
          var_res  RFF-GP leverage of this residue's own embedding (epistemic; needs the spectral norm)
        Invalid positions are zeroed. Call with projected residues (use .project first)."""
        sim = torch.bmm(xa.float(), xb.float().transpose(1, 2))
        neg = torch.finfo(sim.dtype).min
        simb = sim.masked_fill(~mb[:, None, :], neg)
        top2 = simb.topk(2, dim=2).values if simb.shape[2] >= 2 else None
        best = simb.max(dim=2).values
        margin = (top2[..., 0] - top2[..., 1]) if top2 is not None else torch.zeros_like(best)
        mean_own = (best * ma).sum(1, keepdim=True) / ma.sum(1, keepdim=True).clamp_min(1)
        rep = {"best": best * ma, "sel_gap": (best - mean_own) * ma, "margin": margin * ma}
        if get_var:
            B, L, _ = xa.shape
            _, v = self.res_gp(xa.reshape(B * L, -1), get_var=True)
            rep["var_res"] = v.reshape(B, L) * ma
        return rep

    @torch.no_grad()
    def accumulate_residue_precision(self, xa, ma):
        """Add this batch's valid residues to the residue GP's precision. OUTSIDE DDP's forward."""
        x = xa[ma.bool()]
        if x.numel():
            self.res_gp.accumulate(self.res_gp.rff(x))

    @torch.no_grad()
    def fit_all_covariances(self, reduce=True):
        if self.use_gp:
            self.g.gp.fit_covariance(reduce=reduce)
        self.res_gp.fit_covariance(reduce=reduce)

    @torch.no_grad()
    def reset_all_precisions(self):
        if self.use_gp:
            self.g.gp.reset_precision()
        self.res_gp.reset_precision()
