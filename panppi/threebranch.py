"""Three-branch head for a fine-tuned ESM-2 backbone: LoGoPPI's two branches plus our learned-interaction one.

THE ARGUMENT. Decomposing LoGoPPI (job 72773) showed its two branches have OPPOSITE transfer behaviour:
                 global-only    maxsim-only    their 0.5/0.5 blend
    yeast          0.7225         0.6483            0.7464
    ecoli          0.5410         0.7387            0.7389   <- global contributes nothing here
Global is a composition/class prior that transfers eukaryote->eukaryote and dies eukaryote->prokaryote;
MaxSim is residue complementarity and transfers to both. Their fixed blend rides whichever mechanism survives
the domain shift. Crucially NEITHER branch learns an interaction: global sees only mean-pooled vectors, MaxSim
is a dot product and a max. The joint SOTA PPLM-PPI does learn one, residue-level, and beats LoGoPPI by 0.045
on E. coli -- but it cannot cache per protein.

Branch 3 fills that slot: K latent queries read each protein's OWN residues (so it stays cacheable), then one
self-attention block over the concatenated 2K latents lets each protein's latents attend to the partner's.
Learned interaction over compressed representations, with per-protein caching preserved.

WHY THIS NEEDS A FINE-TUNED BACKBONE TO BE A FAIR TEST. Job 73078 trained this branch on LoGoPPI's FROZEN
cached embeddings and it did not help (yeast -0.001, ecoli -0.016). But that space was fine-tuned to serve
THEIR two branches, so the branch could only read someone else's representation, never shape its own. Under
joint fine-tuning the encoder adapts to all three. That is the difference this run exists to measure.

BLEND WEIGHTS ARE FIXED AND EQUAL, deliberately. LoGoPPI hard-codes 0.5/0.5 and validate_calibration RAISES
if the weights differ. That is not arbitrary: because MaxSim is the WORST branch in-distribution (human val
AUPR 0.7768 vs global's 0.9494) and the BEST on transfer, any weighting fitted on in-distribution validation
would downweight precisely the branch that generalises. Fixed weights act as transfer regularisation.

CORRECTED 2026-11-10 (run D rebuild). The earlier version of this file said LoGoPPI trains a MaxSim adapter in a
second stage, and so trained a learned MaxSim (adapter + scale + bias) jointly with its own BCE. That was WRONG.
Their released pipeline (ext/LoGoPPI/scripts/training.py, docstring steps 1-4) trains ONLY the Global loss; MaxSim
is parameter-free symmetric_maxsim on the projected residues of the Global-trained encoder, standardised with
train statistics and affinely calibrated on a held-out human split afterwards. MaxSimStep2Model/MaxSimAdapter
exist in logobert/ but no script uses them, and utils/prediction.py:321 scores with plain symmetric_maxsim.
This matters scientifically: MaxSim may transfer BECAUSE it is never fitted to human labels.
So now: MaxSim has no parameters and no loss; the encoder is trained by the Global loss plus (our addition) the
latent branch's loss; every branch is calibrated post hoc exactly as fit_calibration does.

NO GP/SNGP HEAD HERE. The uncertainty layer is deferred: model.py:241-243 accumulates the GP precision matrix
only in the final epoch, which assumes a settled feature extractor, and that assumption is weaker when the
backbone moves throughout training. Shipping unvalidated uncertainty would undercut the one thing that
distinguishes this project, so this run answers the accuracy question only and the GP head is added back once
the precision-accumulation question is examined. Plain linear heads are used instead.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


class GlobalBranch(nn.Module):
    """Mean-pool each protein, classify [a, b, |a-b|]. Symmetrised by averaging both orderings' logits,
    which is LoGoPPI's global_symmetry: mean_ordered_logits."""

    def __init__(self, d, hid=512):
        super().__init__()
        self.ln = nn.LayerNorm(3 * d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, 1))

    def one(self, a, b):
        return self.mlp(self.ln(torch.cat([a, b, (a - b).abs()], dim=-1))).squeeze(-1)

    def forward(self, pa, pb):
        return 0.5 * (self.one(pa, pb) + self.one(pb, pa))


def symmetric_maxsim(xa, ma, xb, mb):
    """Parameter-free symmetric MaxSim, as logobert/maxsim.py symmetric_maxsim: raw dot products in FP32, each
    residue's best partner residue, averaged over own residues, mean of both directions. Returns RAW scores
    (uncalibrated); calibration happens after training."""
    sim = torch.bmm(xa.float(), xb.float().transpose(1, 2))
    neg = torch.finfo(sim.dtype).min
    best_a = sim.masked_fill(~mb[:, None, :], neg).max(dim=2).values
    best_b = sim.masked_fill(~ma[:, :, None], neg).max(dim=1).values
    sa = (best_a * ma).sum(1) / ma.sum(1).clamp_min(1)
    sb = (best_b * mb).sum(1) / mb.sum(1).clamp_min(1)
    return (sa + sb) * 0.5


# ---- V8 latent branch, mirrored from tunaplus/variants.py + model.py (TUnA layers), WITHOUT the GP head ----
# Self-contained copies so cluster 2 needs no TUnA utils/lookahead imports. Same maths, same spectral norm,
# same mask convention (multiplicative 0/1 mask, masked_fill(mask == 0, -1e10)).
from torch.nn.utils import spectral_norm  # noqa: E402


class TSelfAttention(nn.Module):
    """model.SelfAttention."""

    def __init__(self, hid, heads, dropout):
        super().__init__()
        self.hid, self.heads = hid, heads
        self.w_q, self.w_k = spectral_norm(nn.Linear(hid, hid)), spectral_norm(nn.Linear(hid, hid))
        self.w_v, self.fc = spectral_norm(nn.Linear(hid, hid)), spectral_norm(nn.Linear(hid, hid))
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid // heads)

    def forward(self, q, k, v, mask=None):
        b, dh = q.shape[0], self.hid // self.heads
        Q = self.w_q(q).view(b, -1, self.heads, dh).permute(0, 2, 1, 3)
        K = self.w_k(k).view(b, -1, self.heads, dh).permute(0, 2, 1, 3)
        V = self.w_v(v).view(b, -1, self.heads, dh).permute(0, 2, 1, 3)
        e = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)
        x = torch.matmul(self.do(F.softmax(e, dim=-1)), V)
        return self.fc(x.permute(0, 2, 1, 3).contiguous().view(b, -1, self.hid))


class TFeedforward(nn.Module):
    """model.Feedforward (swish)."""

    def __init__(self, hid, ff, dropout):
        super().__init__()
        self.fc_1, self.fc_2 = spectral_norm(nn.Linear(hid, ff)), spectral_norm(nn.Linear(ff, hid))
        self.do, self.act = nn.Dropout(dropout), nn.SiLU()

    def forward(self, x):
        return self.fc_2(self.do(self.act(self.fc_1(x))))


class TEncoderLayer(nn.Module):
    """model.EncoderLayer: post-LN self-attention + feedforward."""

    def __init__(self, hid, heads, ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(hid), nn.LayerNorm(hid)
        self.do1, self.do2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.sa, self.ff = TSelfAttention(hid, heads, dropout), TFeedforward(hid, ff, dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.do1(self.sa(x, x, x, mask)))
        return self.ln2(x + self.do2(self.ff(x)))


class TCrossBlock(TEncoderLayer):
    """variants.CrossBlock: same layer with separate query and key/value (the latent reader)."""

    def forward(self, q, kv, mask):
        q = self.ln1(q + self.do1(self.sa(q, kv, kv, mask)))
        return self.ln2(q + self.do2(self.ff(q)))


class TCrossMix(nn.Module):
    """variants.CrossMix: attention-only, partner's latents only (block-off-diagonal mask), own residual + LN."""

    def __init__(self, hid, heads, dropout):
        super().__init__()
        self.ln, self.do, self.sa = nn.LayerNorm(hid), nn.Dropout(dropout), TSelfAttention(hid, heads, dropout)

    def forward(self, x, mask):
        return self.ln(x + self.do(self.sa(x, x, x, mask)))


class LatentBranch(nn.Module):
    """V8 (builder.VARIANTS['V8'], config_xspecies.yaml): per protein, TUnA's intra encoder (spectral-normed
    Linear d_in->256 + 1 EncoderLayer over its own residues), then K=16 fixed latents read it (CrossBlock) --
    all cacheable per protein. Pair step = ONE exchange with SPLIT attention: EncoderLayer over the 2K latents with
    a block-diagonal (own-latents) mask, then CrossMix with a block-off-diagonal (partner-only) mask. Each protein's
    own K latents are mean-pooled, the two vectors combined by element-wise max (V8's head='max'), symmetric in
    A/B. V8's GP/SNGP layer is replaced by a plain Linear(256, 1) logit (uncertainty deferred, see module doc).
    Input here is the shared 512-d LoGoPPI projection rather than frozen ESM-150M residues."""

    def __init__(self, d_in, d=256, k=16, heads=8, ff=1024, dropout=0.2):
        super().__init__()
        self.ft = spectral_norm(nn.Linear(d_in, d))
        self.intra = TEncoderLayer(d, heads, ff, dropout)
        self.latents = nn.Parameter(torch.randn(k, d))            # as variants.py (std 1)
        self.reader = TCrossBlock(d, heads, ff, dropout)
        self.own_layer = TEncoderLayer(d, heads, ff, dropout)
        self.cross = TCrossMix(d, heads, dropout)
        self.out = nn.Linear(d, 1)
        self.k = k

    def read(self, x, m):
        """Per-protein, cacheable: [B, L, d_in] residues -> [B, K, d] latents."""
        mm = m.to(x.dtype)
        h = self.intra(self.ft(x), mm[:, None, :, None] * mm[:, None, None, :])   # TUnA make_masks: [B,1,L,L]
        q = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.reader(q, h, mm[:, None, None, :])

    def exchange(self, la, lb):
        k = self.k
        z = torch.cat([la, lb], dim=1)
        blk = torch.cat([torch.zeros(k, device=z.device), torch.ones(k, device=z.device)])
        same = (blk[:, None] == blk[None, :]).to(z.dtype)[None, None]
        z = self.own_layer(z, same)            # within-protein latent attention (+ feedforward)
        z = self.cross(z, 1.0 - same)          # partner-only attention
        return torch.maximum(z[:, :k].mean(1), z[:, k:].mean(1))

    def forward(self, xa, ma, xb, mb):
        return self.out(self.exchange(self.read(xa, ma), self.read(xb, mb))).squeeze(-1)


def hf_init(module, std=0.02):
    """EsmPreTrainedModel._init_weights, which LoGoPPI applies to its projection, LayerNorm and Global MLP
    (logobert/esm2_global.py:_initialize_added_modules): normal(0, 0.02) weights, zero bias, LayerNorm 1/0."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ThreeBranchHead(nn.Module):
    """Projection (Linear d_enc->512 + dropout 0.1) shared by all branches, then:
      global  - trained (BCE), LoGoPPI architecture and init
      maxsim  - parameter-free on the projected residues, NO loss (as LoGoPPI)
      latent  - trained (BCE), our addition
    forward returns raw per-branch outputs only; blending happens after post-hoc calibration."""

    def __init__(self, d_enc, d_proj=512, latent_dim=256, k=16, use_latent=True):
        super().__init__()
        self.proj = nn.Linear(d_enc, d_proj)
        self.drop = nn.Dropout(0.1)
        self.g = GlobalBranch(d_proj)
        self.l = LatentBranch(d_proj, d=latent_dim, k=k) if use_latent else None
        for mod in (self.proj, self.g):
            mod.apply(hf_init)

    def project(self, h):
        return self.drop(self.proj(h))

    def score_projected(self, xa, ma, xb, mb, with_maxsim=True):
        """{branch: raw logit/score} from projected residues. Used in training (with_maxsim=False: it has no
        loss, so computing it would only cost time) and at scoring time from the per-protein cache."""
        out = {"global": self.g(masked_mean(xa, ma), masked_mean(xb, mb))}
        if with_maxsim:
            out["maxsim"] = symmetric_maxsim(xa, ma, xb, mb)
        if self.l is not None:
            out["latent"] = self.l(xa, ma, xb, mb)
        return out

    def forward(self, ha, ma, hb, mb, with_maxsim=True):
        return self.score_projected(self.project(ha), ma, self.project(hb), mb, with_maxsim)
