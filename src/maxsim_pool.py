"""Sparse (sparsemax) pooling for MaxSim -- BUILT AND TESTED 2026-11-12, NOT USED BY ANY RUN.

Deliberately not in E3: E3's maxsim_frozen must stay LoGoPPI's exact construction, because that branch is the
one whose robustness the design exploits (in run D it held 0.763 on E. coli across nine epochs while Global
fell 0.7172 -> 0.4966). Changing its pooling is the single modification that most directly risks that.

KEPT because the DB5.5 results give it a clear purpose: MaxSim's per-residue attribution loses to an
amino-acid composition lookup (0.5447 vs 0.5833) and the pooled MEAN is the mechanism -- optimising a mean
lets the model spread signal rather than localise it (XTR, arXiv 2304.01982). Sparse pooling is therefore the
natural INTERPRETABILITY intervention, not an accuracy play: the sparse-attention literature reports
accuracy-neutral results (entmax "does not hurt performance in any of the datasets"; sparsemax 26.4 vs 26.9
BLEU in one MT study).

tau -> inf reproduces LoGoPPI's masked mean EXACTLY, so the null sits inside the hypothesis class and a tau
that stays large is itself an interpretable negative result. sparsemax (alpha=2) rather than learnable-alpha
entmax: `entmax` is not installed in ppi_env and general alpha needs a custom backward, whereas sparsemax is
closed-form via sort+cumsum with autograd correct a.e.

Tests: tunaplus/tests/test_sparse_pool.py, 17/17 CPU checks. Check 10 records the finding that killed the
length-asymmetry claim: sparse pooling cures DILUTION on the long protein's own residues (drift 0.82 -> 0.33)
but AMPLIFIES sqrt(2 ln L) max-inflation on the short protein's side (0.28 -> 0.33). It trades one length
artefact for another and must NOT be sold as a peptide fix.
"""
import torch

from threebranch import symmetric_maxsim

# WHY. symmetric_maxsim takes max over the PARTNER's residues (right: "who does my residue bind?") and then a
# MEAN over ALL of its own residues (wrong: most residues are not interface). For a 300-residue protein with
# ~20 interface residues that is ~93% dilution, and it is the source of the measured length dependence
# (max-mean)/sigma ~ sqrt(2 ln N) (jobs 73000/73044). The same train/inference mismatch is documented in
# retrieval: XTR (arXiv 2304.01982) found ColBERT "assigns high scores to irrelevant tokens because it only
# optimizes for the final document-level sum" and fixed it by masking non-top-k token scores DURING TRAINING.
#
# WHAT. Self-weighted sparsemax pooling: w = sparsemax(best / tau) over own valid residues, score = sum(w*best).
# tau is ONE learnable scalar (stored as log_tau), shared by both directions so the pair stays order invariant.
#   tau -> inf  : w -> uniform  => EXACTLY LoGoPPI's masked mean (the null this must beat)
#   tau -> 0    : w -> one-hot  => pure max
#   in between  : sparsemax gives EXACT ZEROS, so the number of contributing residues is learned, not set.
# The support size |{w>0}| is readable per protein => "K" is an output, comparable with Propedia's annotated
# interface residue counts (the same ground truth validated by AA composition for the V9 K(L) fit).
#
# WHY sparsemax (alpha=2) AND NOT learnable-alpha entmax, which was the first sketch: general alpha-entmax needs
# bisection plus a custom backward for d/d(alpha), and `entmax` is NOT installed in ppi_env (checked 2026-11-12).
# sparsemax has a closed form via sort+cumsum whose autograd gradient is correct almost everywhere (the support
# is locally constant), so it is ~15 lines with no custom autograd. tau already moves the effective K
# continuously, which is what "learn K" needs; alpha would only change the shape of the taper.
#
# CACHEABILITY IS UNCHANGED: this acts on `best`, computed at pair time from per-protein cached residue
# projections. Nothing here makes a protein's representation depend on its partner.


def _sparsemax(z, mask):
    """sparsemax(z) over the last dim, restricted to mask (True = valid); invalid positions get weight 0.
    Euclidean projection onto the simplex (Martins & Astudillo 2016): sort descending, k(z) = max{k : 1 + k*z_k
    > cumsum_k}, tau = (cumsum_k - 1)/k, p = [z - tau]_+. sparsemax is shift invariant, so z is centred on its
    masked max first for numerical safety."""
    z = z.float()
    zmax = z.masked_fill(~mask, float("-inf")).max(dim=-1, keepdim=True).values
    z = (z - zmax.clamp_min(-1e30)).masked_fill(~mask, -1e9)
    zs, _ = torch.sort(z, dim=-1, descending=True)          # masked entries sort to the end (-1e9)
    n = mask.sum(-1, keepdim=True)                          # valid count per row
    rng = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype).expand_as(z)
    valid = rng <= n
    cs = zs.masked_fill(~valid, 0.0).cumsum(-1)
    k = ((1 + rng * zs > cs) & valid).to(z.dtype).sum(-1, keepdim=True).clamp_min(1.0)
    tau = (cs.gather(-1, k.long() - 1) - 1.0) / k
    return torch.clamp(z - tau, min=0.0).masked_fill(~mask, 0.0)


def sparse_pool(vals, mask, log_tau):
    """Self-weighted sparsemax pooling of `vals` [B, L] over `mask` [B, L]. Returns [B]."""
    tau = log_tau.float().exp().clamp(1e-2, 1e4)
    w = _sparsemax(vals.float() / tau, mask)
    return (w * vals.float()).sum(-1)


def symmetric_maxsim_pooled(xa, ma, xb, mb, log_tau=None):
    """symmetric_maxsim with the own-residue mean replaced by sparsemax pooling. log_tau=None reproduces
    symmetric_maxsim EXACTLY (asserted in tests), so one code path serves E1 and E2."""
    if log_tau is None:
        return symmetric_maxsim(xa, ma, xb, mb)
    sim = torch.bmm(xa.float(), xb.float().transpose(1, 2))
    neg = torch.finfo(sim.dtype).min
    best_a = sim.masked_fill(~mb[:, None, :], neg).max(dim=2).values
    best_b = sim.masked_fill(~ma[:, :, None], neg).max(dim=1).values
    sa = sparse_pool(best_a, ma.bool(), log_tau)
    sb = sparse_pool(best_b, mb.bool(), log_tau)
    return (sa + sb) * 0.5


def maxsim_support(xa, ma, xb, mb, log_tau):
    """DIAGNOSTIC, not used in the score: per-protein learned K (count of residues with w > 0) and the pooling
    weights themselves, for (a) the Propedia interface-residue test and (b) the uncertainty signal."""
    sim = torch.bmm(xa.float(), xb.float().transpose(1, 2))
    neg = torch.finfo(sim.dtype).min
    best_a = sim.masked_fill(~mb[:, None, :], neg).max(dim=2).values
    wa = _sparsemax(best_a.float() / log_tau.float().exp().clamp(1e-2, 1e4), ma.bool())
    return (wa > 0).sum(-1), wa


