"""CPU tests for analysis/posthoc_uncertainty.py (the post-hoc GP add-on for a plain-head model).

ASSERTED: the identity the whole add-on rests on (mlp[3](z) == the model's own Global logit) through the REAL main_e3 Model with a fake ESM;
the leverage GP is distance-aware on a synthetic mixture AND passes its own negative control; ECE, the mean-field scalar, error-detection
AUROC and F1-at-coverage behave correctly on synthetic cases where the right answer is known (informative variance vs noise).
NOT ASSERTED, deliberately: that the variance of a real model is informative -- that is what the readouts U1-U5 measure on real data.
Run: python -I tunaplus/tests/test_posthoc_uncertainty.py
"""
import contextlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
import main_e3 as M  # noqa: E402
import posthoc_uncertainty as P  # noqa: E402

torch.manual_seed(0)
OK = []
DEV = torch.device("cpu")


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""), flush=True)
    OK.append(bool(cond))


class FakeTok:
    def __call__(self, seqs, return_tensors=None, padding=True, truncation=True, max_length=None):
        ids = [[1] + [2 + ord(c) % 20 for c in s] + [3] for s in seqs]
        L = max(len(x) for x in ids)
        ii = torch.zeros(len(ids), L, dtype=torch.long)
        am = torch.zeros_like(ii)
        for i, x in enumerate(ids):
            ii[i, :len(x)] = torch.tensor(x)
            am[i, :len(x)] = 1
        return {"input_ids": ii, "attention_mask": am}


class FakeEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb, self.lin = nn.Embedding(30, d), nn.Linear(d, d)

    def forward(self, input_ids, attention_mask):
        class O:
            pass
        o = O()
        o.last_hidden_state = self.lin(self.emb(input_ids)) * attention_mask.unsqueeze(-1)
        return o


# ---- A. the identity, through the real plain-head Model (the E3a / CTRL flags)
torch.manual_seed(1)
model = M.Model("unused", 32, 60, grad_ckpt=False, use_latent=False, encoder=FakeEnc(32), tok=FakeTok(), use_sn=False,
                use_gp=False, seed=1, use_res_gp=False)
model.eval()
rng = np.random.default_rng(3)
AA = "ACDEFGHIKLMNPQRSTVWY"
seqs = {f"p{i}": "".join(rng.choice(list(AA), int(rng.integers(20, 45)))) for i in range(60)}
ids = sorted(seqs)
pairs = pd.DataFrame({"a": rng.choice(ids, 100), "b": rng.choice(ids, 100), "label": (rng.random(100) < 0.3).astype(int)})
cache = M.embed_cache(model, set(pairs.a) | set(pairs.b), seqs, DEV, contextlib.nullcontext)
raw = M.score_cached(model, pairs, cache, DEV)
Z = P.pair_features(model.head, cache, pairs, DEV)
lg = model.head.g.mlp[3](Z).squeeze(-1).detach().numpy()
check("U0 identity: mlp[3](z) equals the model's own Global logit (real Model, plain head)",
      float(np.abs(lg - raw["global"]).max()) < 1e-4, f"max |delta| {float(np.abs(lg - raw['global']).max()):.2e}")
pooled = P.embed_pooled(model, set(pairs.a) | set(pairs.b), seqs, DEV, contextlib.nullcontext)
Zp = P.pair_features(model.head, pooled, pairs, DEV)

# ---- A2. a cache that MIXES devices (the 2026-11-17 GPU crash): residues on cuda, pooled vectors on cpu. Needs a GPU, so it only
# runs inside the GPU allocation; locally it is skipped (and says so) instead of silently passing.
if torch.cuda.is_available():
    cuda = torch.device("cuda")
    ks = sorted(set(pairs.a) | set(pairs.b))
    mixed = {k: (cache[k].to(cuda) if i % 2 == 0 else cache[k].float().mean(0, keepdim=True).cpu()) for i, k in enumerate(ks)}
    Zm = P.pair_features(model.head.to(cuda), mixed, pairs, cuda)
    model.head.to(DEV)
    check("pair_features accepts a cache that mixes cuda and cpu tensors", torch.allclose(Zm.float(), Z.float(), atol=1e-3),
          f"max |delta| {float((Zm.float() - Z.float()).abs().max()):.2e}")
else:
    print("SKIP  pair_features mixed-device cache check (no CUDA on this machine; it runs inside the GPU allocation)", flush=True)
check("embed_pooled (pooled-only, memory-light) gives the SAME z as the full residue cache",
      float((Zp - Z).abs().max()) < 1e-5, f"max |delta| {float((Zp - Z).abs().max()):.2e}")
check("z has the head's hidden width", Z.shape == (100, model.head.g.mlp[0].out_features))

# ---- B. the leverage GP: distance-aware on a mixture, and its own negative control
g = torch.Generator().manual_seed(2)
d = 64
centres = torch.randn(6, d, generator=g) * 3.0
Xtr = centres[torch.randint(0, 6, (4000,), generator=g)] + 0.6 * torch.randn(4000, d, generator=g)
Xin = centres[torch.randint(0, 6, (1000,), generator=g)] + 0.6 * torch.randn(1000, d, generator=g)       # held-out, same dist
Xout = (torch.randn(6, d, generator=g) * 3.0)[torch.randint(0, 6, (1000,), generator=g)] + 0.6 * torch.randn(1000, d, generator=g)
gp, amp = P.fit_leverage_gp(Xtr, DEV, rffs=512, seed=1)
v_tr, v_in, v_out = (P.gp_variance(gp, X, DEV) for X in (Xtr[:1000], Xin, Xout))
check("amplitude is calibrated by the median heuristic (not left at 1.0)", abs(amp - 1.0) > 1e-3, f"amp {amp:.4f}")
check("U1 negative control: held-out in-distribution variance is within ~3x of the fitted points",
      np.median(v_in) / np.median(v_tr) < 3.0, f"{np.median(v_in) / np.median(v_tr):.2f}x")
check("U2 distance awareness: out-of-distribution variance clearly exceeds in-distribution",
      np.median(v_out) > 3.0 * np.median(v_in), f"{np.median(v_out) / np.median(v_in):.1f}x")
lab = np.r_[np.zeros(1000, bool), np.ones(1000, bool)]
check("AUROC of variance for in-dist vs OOD is high", M.auroc(lab, np.r_[v_in, v_out]) > 0.9,
      f"{M.auroc(lab, np.r_[v_in, v_out]):.3f}")

# ---- C. ECE
r = np.random.default_rng(0)
p = r.random(20000)
y = r.random(20000) < p
check("ECE ~ 0 for a calibrated probability", P.ece(y, p) < 0.02, f"{P.ece(y, p):.4f}")
check("ECE is large for a miscalibrated probability", P.ece(y, p * 0.3) > 0.1, f"{P.ece(y, p * 0.3):.4f}")
check("ECE handles a rare-positive base rate (equal-mass bins)", P.ece(r.random(20000) < 0.05, np.full(20000, 0.05) + 0.0 * r.random(20000)) < 0.02)

# ---- D. mean-field scalar: informative variance gets s > 0 and lowers NLL; noise variance does not
n = 20000
z = r.standard_normal(n) * 2.5
v = r.random(n) * 3.0
y_inf = r.random(n) < P.sigmoid(z / np.sqrt(1 + 2.0 * v))       # labels generated WITH shrinkage by the variance
s1, n1, n0 = P.fit_mf_scale(z, v, y_inf.astype(float))
check("mean-field scalar recovers a clearly positive s when the variance is informative", 0.8 < s1 < 5.0 and n1 < n0 - 0.005,
      f"s {s1:.2f} NLL {n0:.4f} -> {n1:.4f}")
y_noise = r.random(n) < P.sigmoid(z)                            # variance irrelevant
s2, n2, n0b = P.fit_mf_scale(z, r.random(n) * 3.0, y_noise.astype(float))
check("mean-field scalar stays ~0 and gains ~nothing when the variance is noise", s2 < 0.15 and (n0b - n2) < 0.003,
      f"s {s2:.3f} NLL gain {n0b - n2:.4f}")

# ---- E. error detection and selective prediction
pr = P.sigmoid(z)
yb = r.random(n) < pr
thr = P.best_threshold(yb, pr)
err = (pr >= thr) != yb
v_good = err * 2.0 + r.random(n)                                # variance high exactly where the model errs
v_rand = r.random(n)
check("U3 error AUROC is high when the variance tracks the errors", P.error_auroc(yb, pr, v_good, thr) > 0.8,
      f"{P.error_auroc(yb, pr, v_good, thr):.3f}")
check("U3 error AUROC is ~0.5 for a variance that is noise", abs(P.error_auroc(yb, pr, v_rand, thr) - 0.5) < 0.03,
      f"{P.error_auroc(yb, pr, v_rand, thr):.3f}")
fv, fr = P.f1_at_coverage(yb, pr, v_good, thr, 0.5, np.random.default_rng(1))
check("U4 rejecting by an informative variance beats random rejection", fv > fr + 0.05, f"{fv:.3f} vs random {fr:.3f}")
fv2, fr2 = P.f1_at_coverage(yb, pr, v_rand, thr, 0.5, np.random.default_rng(1))
check("U4 rejecting by a noise variance does NOT beat random (|diff| small)", abs(fv2 - fr2) < 0.03, f"{fv2:.3f} vs {fr2:.3f}")

print(f"\n{sum(OK)}/{len(OK)} checks pass", flush=True)
sys.exit(0 if all(OK) else 1)
