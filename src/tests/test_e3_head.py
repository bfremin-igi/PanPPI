"""CPU tests for E3's two-projection head (threebranch_e3.E3Head).

The tests that matter are the GRADIENT ISOLATION ones. The entire architecture rests on three claims about
where gradients go, and every one of them fails SILENTLY if wired wrong -- the model would still train, still
produce numbers, and would just quietly be E1 again with an extra unused Linear:
  1. the rank loss reaches the ENCODER            (that is the point: it is there to improve Global)
  2. the rank loss NEVER reaches proj_g           (or maxsim_frozen stops being LoGoPPI's construction)
  3. the Global loss NEVER reaches proj_m
Run: python -I tunaplus/tests/test_e3_head.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threebranch import symmetric_maxsim  # noqa: E402
from threebranch_e3 import E3Head  # noqa: E402

torch.manual_seed(0)
OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


B, La, Lb, D = 3, 23, 17, 64
head = E3Head(D, d_proj=32, rffs=64, use_gp=True, sn_on_projg=True, seed=3)
ha, hb = torch.randn(B, La, D), torch.randn(B, Lb, D)
ma = torch.zeros(B, La, dtype=torch.bool)
mb = torch.zeros(B, Lb, dtype=torch.bool)
for i in range(B):
    ma[i, : La - 4 * i] = True
    mb[i, : Lb - 3 * i] = True

head.eval()
out = head(ha, ma, hb, mb)
check("returns global, maxsim (frozen) and maxsim_trained",
      {"global", "maxsim", "maxsim_trained"} <= set(out))
check("the two MaxSims are genuinely different branches",
      not torch.allclose(out["maxsim"], out["maxsim_trained"], atol=1e-4))
check("maxsim (frozen) is computed on proj_g",
      torch.allclose(out["maxsim"],
                     symmetric_maxsim(head.project_g(ha), ma, head.project_g(hb), mb), atol=1e-5))
check("ORDER INVARIANT on every branch",
      all(torch.allclose(out[k], head(hb, mb, ha, ma)[k], atol=1e-5)
          for k in ("global", "maxsim", "maxsim_trained")))

pad = 6
ha2 = torch.cat([ha, torch.randn(B, pad, D) * 40], 1)
ma2 = torch.cat([ma, torch.zeros(B, pad, dtype=torch.bool)], 1)
o2 = head(ha2, ma2, hb, mb)
check("PADDING INVARIANT on every branch",
      all(torch.allclose(out[k], o2[k], atol=1e-3) for k in ("global", "maxsim", "maxsim_trained")))


# ---- GRADIENT ISOLATION -------------------------------------------------------------------------
def grads_from(loss_fn):
    """Zero every grad, run one backward, report which parameter groups received gradient."""
    head.zero_grad(set_to_none=True)
    h = ha.clone().requires_grad_(True)
    loss_fn(h).backward()
    got = lambda mod: any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    return {"encoder_input": h.grad is not None and h.grad.abs().sum() > 0,
            "proj_g": got(head.proj_g), "proj_m": head.proj_m.weight.grad is not None
            and head.proj_m.weight.grad.abs().sum() > 0, "global_head": got(head.g)}


# a rank-loss stand-in: a MaxSim-shaped objective on the proj_m residues
def rank_like(h):
    xma, xmb = head.rank_inputs(h, hb)
    return symmetric_maxsim(xma, ma, xmb, mb).mean()


g_rank = grads_from(rank_like)
check("1. rank loss REACHES the encoder (its whole purpose)", g_rank["encoder_input"])
check("2. rank loss does NOT touch proj_g", not g_rank["proj_g"])
check("   rank loss does train proj_m", g_rank["proj_m"])
check("   rank loss does NOT touch the Global head", not g_rank["global_head"])


def global_like(h):
    return head(h, ma, hb, mb, with_maxsim=False, with_trained_maxsim=False)["global"].mean()


g_glob = grads_from(global_like)
check("3. Global loss does NOT touch proj_m", not g_glob["proj_m"])
check("   Global loss trains proj_g and the Global head",
      g_glob["proj_g"] and g_glob["global_head"])
check("   Global loss reaches the encoder", g_glob["encoder_input"])

# the frozen branch must never be a gradient path for a MaxSim objective
head.zero_grad(set_to_none=True)
h = ha.clone().requires_grad_(True)
head.score_projected(head.project_g(h), ma, head.project_g(hb), mb)["maxsim"].mean().backward()
check("   (reference) a MaxSim loss on proj_g WOULD train proj_g -- which is why the rank loss must not",
      any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.proj_g.parameters()))

# ---- uncertainty still works -----------------------------------------------------------------------
pa, pb = head.project_g(ha), head.project_g(hb)
head.reset_all_precisions()
head.g(pa.mean(1), pb.mean(1), update_precision=True)
head.g.gp.fit_covariance(reduce=False)
o = head.score_projected(pa, ma, pb, mb, get_var=True)
check("pair variance + mean-field probability produced",
      {"global_var", "global_prob_mf"} <= set(o) and torch.isfinite(o["global_var"]).all())
check("mean-field output is a probability",
      bool(((o["global_prob_mf"] > 0) & (o["global_prob_mf"] < 1)).all()))
check("residue GP is OFF by default (DB5.5: residue attribution loses to a composition lookup)",
      head.res_gp is None)

sv = float(torch.linalg.matrix_norm(head.proj_g.weight.detach(), ord=2))
head.train()
for _ in range(50):
    head.project_g(ha)
head.eval()
check("spectral norm on proj_g converges to 1 after the power iteration",
      abs(float(torch.linalg.matrix_norm(head.proj_g.weight.detach(), ord=2)) - 1.0) < 0.02)

plain = E3Head(D, d_proj=32, use_gp=True, sn_on_projg=False, seed=3)
check("sn_on_projg=False leaves proj_g an unconstrained Linear", not hasattr(plain.proj_g, "lin"))

print(f"\n{sum(OK)}/{len(OK)} checks pass")
sys.exit(0 if all(OK) else 1)
