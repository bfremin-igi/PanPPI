"""E3 head: TWO projections off ONE encoder, so the rank loss can improve Global WITHOUT destroying the
parameter-free MaxSim -- plus E2's spectral-norm + GP uncertainty.

THE IDEA (user, 2026-11-12): "it doesn't matter that maxsim learns, it still roughly converges on global and
improves global even... then the untrained maxsim will still likely do well on E. coli". That reframes the
trained MaxSim as a TRAINING DEVICE rather than a scoring branch. Its measured value is not its own score, it
is what it does to the shared encoder:
    E. coli Global AUPR   LoGoPPI 0.5829 | run D (no rank loss) 0.7172 ep4 -> 0.5487 ep11, COLLAPSING
                          E1 (rank loss) 0.7070-0.7164, no collapse
while the trained MaxSim itself converges onto Global (human gap 0.002, vs LoGoPPI's 0.175) and, on DB5.5,
becomes ANTI-localised (+0.71 to +2.00 A farther from the partner than exposure-matched random, CIs excluding
0) where LoGoPPI's untrained MaxSim stays localised (-0.85 A [-1.88, -0.18]).

So: keep the rank loss for what it does to the encoder, and keep a SEPARATE parameter-free MaxSim for scoring.

  proj_g  trained by the GLOBAL BCE ONLY (no MaxSim gradient ever) -> Global branch AND maxsim_frozen.
          That is LoGoPPI's exact construction: their MaxSim is parameter-free on a globally-trained
          projection, never fitted by a MaxSim objective.
  proj_m  trained by the RANK LOSS -> maxsim_trained. Gradient flows proj_m -> ENCODER on purpose: that is
          the whole point, the encoder is what we want the rank loss to improve.

SCORING STAYS LoGoPPI's 0.5 global + 0.5 maxsim_frozen -- NO new blend weights. threebranch.py:25 records why
that matters: MaxSim is the WORST branch in-distribution and the BEST on transfer, so any weighting fitted on
human validation downweights precisely the branch that generalises. maxsim_trained is still RETURNED and
logged so the 3-branch option can be scored afterwards as a sensitivity check, never as the headline.

THE OPEN QUESTION THIS EXISTS TO ANSWER. maxsim_frozen reads an encoder that the rank loss HAS shaped, so it
is not perfectly insulated. Whether LoGoPPI's localisation survives a shared encoder is unknown, and E1 cannot
tell us because it changed encoder and projection together. The readout already exists: run the DB5.5
localisation test on maxsim_frozen. Near -0.85 A => the property survives and one cacheable model carries both
capabilities. Anti-localised like E1 => the damage is in the ENCODER, and only independent encoders can work.
Either answer is worth the run.

UNCERTAINTY: as E2 (threebranch_e2) -- spectral norm + an RFF-GP on the Global branch for a distance-aware
pair-level variance. The residue-level GP is DEFAULT OFF here: DB5.5 showed per-residue MaxSim attribution is
worse than an amino-acid composition lookup (0.5447 vs 0.5833), so a residue variance would be uncertainty
about noise. It can be switched on for diagnostics, not for claims.
"""
import torch
from torch import nn

from threebranch import hf_init, masked_mean, symmetric_maxsim
from threebranch_e2 import RFFGP, GlobalGPBranch, ScaledSN, mean_field


class E3Head(nn.Module):
    """Two projections, three scores, one encoder.

    sn_on_projg: spectral-normalise proj_g. Needed for the GP's distance awareness to mean anything, but it
    also constrains the space maxsim_frozen lives in, which is a deviation from LoGoPPI. Exposed as a flag so
    the ablation is possible; GlobalGPBranch spectral-normalises its own body regardless, so the pair GP keeps
    some distance awareness even with this off."""

    def __init__(self, d_enc, d_proj=512, rffs=1024, res_rffs=256, use_gp=True,
                 sn_on_projg=True, sn_bound=1.0, use_res_gp=False, seed=123):
        super().__init__()
        pg, pm = nn.Linear(d_enc, d_proj), nn.Linear(d_enc, d_proj)
        pg.apply(hf_init)
        pm.apply(hf_init)
        self.proj_g = ScaledSN(pg, sn_bound) if sn_on_projg else pg
        self.proj_m = pm
        self.drop = nn.Dropout(0.1)
        self.use_gp = use_gp
        if use_gp:
            self.g = GlobalGPBranch(d_proj, rffs=rffs, seed=seed)
            self.g.ln.apply(hf_init)
        else:
            from threebranch import GlobalBranch
            self.g = GlobalBranch(d_proj)
            self.g.apply(hf_init)
        self.res_gp = RFFGP(d_proj, rffs=res_rffs, likelihood="leverage", seed=seed + 1) if use_res_gp else None
        self.l = None          # compat: main_e*.score_cached tests `head.l is not None`

    # ---- projections -------------------------------------------------------------------------------
    def project_g(self, h):
        """Global / frozen-MaxSim space. Trained by the Global BCE only."""
        return self.drop(self.proj_g(h))

    def project_m(self, h):
        """Trained-MaxSim space. Trained by the rank loss, whose gradient reaches the ENCODER by design."""
        return self.drop(self.proj_m(h))

    def project(self, h):
        """Default projection for callers that cache ONE space (scoring uses proj_g, since the reported
        blend is global + maxsim_frozen)."""
        return self.project_g(h)

    # ---- scoring -----------------------------------------------------------------------------------
    def score_projected(self, xa, ma, xb, mb, with_maxsim=True, update_precision=False, get_var=False,
                        xma=None, xmb=None):
        """xa/xb are proj_g residues. xma/xmb are proj_m residues; pass them to also get maxsim_trained."""
        out = {}
        pa, pb = masked_mean(xa, ma), masked_mean(xb, mb)
        if self.use_gp and get_var:
            logit, var = self.g(pa, pb, update_precision=update_precision, get_var=True)
            out["global"], out["global_var"] = logit, var
            out["global_prob_mf"] = mean_field(logit, var)
        elif self.use_gp:
            out["global"] = self.g(pa, pb, update_precision=update_precision)
        else:
            out["global"] = self.g(pa, pb)
        if with_maxsim:
            # LoGoPPI's construction: parameter-free, on the globally-trained projection, never a MaxSim loss.
            out["maxsim"] = symmetric_maxsim(xa, ma, xb, mb)
            if xma is not None:
                out["maxsim_trained"] = symmetric_maxsim(xma, ma, xmb, mb)
        return out

    def forward(self, ha, ma, hb, mb, with_maxsim=True, update_precision=False, get_var=False,
                with_trained_maxsim=True):
        xa, xb = self.project_g(ha), self.project_g(hb)
        xma = xmb = None
        if with_trained_maxsim:
            xma, xmb = self.project_m(ha), self.project_m(hb)
        return self.score_projected(xa, ma, xb, mb, with_maxsim, update_precision, get_var, xma, xmb)

    def rank_inputs(self, ha, hb):
        """Residues the RANK LOSS must be computed on. Kept as an explicit method so a training loop cannot
        accidentally put the rank loss on proj_g -- which would silently recreate E1 and destroy the frozen
        branch this architecture exists to protect."""
        return self.project_m(ha), self.project_m(hb)

    def residue_report(self, xa, ma, xb, mb, get_var=True):
        """Per-residue diagnostics on the proj_g (frozen-MaxSim) space. `best`, `sel_gap` and `margin` are
        free and always returned; `var_res` only exists when the residue GP is enabled, which it is NOT by
        default -- DB5.5 showed per-residue attribution loses to a composition lookup, so a residue variance
        would be quantifying noise."""
        sim = torch.bmm(xa.float(), xb.float().transpose(1, 2))
        neg = torch.finfo(sim.dtype).min
        simb = sim.masked_fill(~mb[:, None, :], neg)
        top2 = simb.topk(2, dim=2).values if simb.shape[2] >= 2 else None
        best = simb.max(dim=2).values
        margin = (top2[..., 0] - top2[..., 1]) if top2 is not None else torch.zeros_like(best)
        mean_own = (best * ma).sum(1, keepdim=True) / ma.sum(1, keepdim=True).clamp_min(1)
        rep = {"best": best * ma, "sel_gap": (best - mean_own) * ma, "margin": margin * ma}
        if get_var and self.res_gp is not None:
            B, L, _ = xa.shape
            rep["var_res"] = self.res_gp(xa.reshape(B * L, -1), get_var=True)[1].reshape(B, L) * ma
        return rep

    @torch.no_grad()
    def accumulate_residue_precision(self, xa, ma):
        if self.res_gp is not None:
            x = xa[ma.bool()]
            if x.numel():
                self.res_gp.accumulate(self.res_gp.rff(x))

    # ---- uncertainty plumbing (same contract as E2Head) --------------------------------------------
    @torch.no_grad()
    def fit_all_covariances(self, reduce=True):
        if self.use_gp:
            self.g.gp.fit_covariance(reduce=reduce)
        if self.res_gp is not None:
            self.res_gp.fit_covariance(reduce=reduce)

    @torch.no_grad()
    def reset_all_precisions(self):
        if self.use_gp:
            self.g.gp.reset_precision()
        if self.res_gp is not None:
            self.res_gp.reset_precision()
