"""Inference code of the trained PanPPI (E3a) model: encoder + projection head, per-protein embedding cache, cached pair scoring and calibration.
Functions below are copied unchanged from the training/scoring module (main_e3.py) so that checkpoints load strictly and scores match the manuscript's."""
import math

import numpy as np
import torch

from .threebranch_e3 import E3Head


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
