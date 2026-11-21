"""CPU tests for E3a: E3's architecture with the uncertainty machinery OFF (--no-sn --no-gp).

E3a is the ACCURACY CONTROL for E3 (user, 2026-11-12: "we can just try it with uncertainty after the fact ...
not spectral etc but an add-on at the end"). Two things must be true for that plan to work, and both are
asserted here rather than assumed:

  A. Turning the uncertainty off must not disturb the architecture under test. The gradient isolation is the
     whole of E3 and it fails SILENTLY if wired wrong, so all three claims are re-checked with sn/gp off:
     the rank loss reaches the ENCODER and proj_m, never proj_g or the Global head; the Global loss never
     reaches proj_m. With --no-sn --no-gp the Global branch is LoGoPPI's plain construction exactly.

  B. A post-hoc uncertainty must still be FITTABLE on the finished checkpoint. The pair GP replaces the
     Global head's final Linear, so with --no-gp no GP exists at all and E3a's run prints no variance. The
     add-on therefore has to fit an RFF-GP on the trained head's PENULTIMATE features afterwards. The test
     below proves those features are recoverable and, crucially, that the model's own logit is EXACTLY a
     linear map of them (the symmetrisation commutes with the final Linear), so a post-hoc GP sits on the
     same quantity the model actually scored -- it is not a different model.

     NOT claimed here: that such a variance is INFORMATIVE. SNGP's distance awareness rests on the spectral
     norm E3a removes, so whether the post-hoc variance separates near from far is an open question to be
     MEASURED (E3 vs E3a, ECE / selective prediction), not a property to assert in a unit test.

Deliberately a SEPARATE file from test_e3_head.py: that one is run by job 41249819's smoke 0, which re-runs
on a requeue, so it is left untouched while E3 is in flight.
Run: python -I tunaplus/tests/test_e3a_nogp.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threebranch import GlobalBranch, symmetric_maxsim  # noqa: E402
from threebranch_e2 import RFFGP  # noqa: E402
from threebranch_e3 import E3Head  # noqa: E402

torch.manual_seed(0)
OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


B, La, Lb, D = 3, 23, 17, 64
# exactly the E3a run's head configuration: use_gp=False (--no-gp), sn_on_projg=False (--no-sn), res_gp off
head = E3Head(D, d_proj=32, use_gp=False, sn_on_projg=False, use_res_gp=False, seed=3)
ha, hb = torch.randn(B, La, D), torch.randn(B, Lb, D)
ma = torch.zeros(B, La, dtype=torch.bool)
mb = torch.zeros(B, Lb, dtype=torch.bool)
for i in range(B):
    ma[i, : La - 4 * i] = True
    mb[i, : Lb - 3 * i] = True
head.eval()

# ---- A1. the configuration is LoGoPPI's plain construction ---------------------------------------
check("--no-gp gives LoGoPPI's plain GlobalBranch (no RFF-GP anywhere)",
      isinstance(head.g, GlobalBranch) and not hasattr(head.g, "gp"))
check("--no-sn leaves proj_g an unconstrained Linear (LoGoPPI's projection)",
      isinstance(head.proj_g, torch.nn.Linear))
check("proj_m is still a separate unconstrained Linear", isinstance(head.proj_m, torch.nn.Linear)
      and head.proj_m is not head.proj_g)
check("no residue GP", head.res_gp is None)

out = head(ha, ma, hb, mb)
check("returns global, maxsim (frozen) and maxsim_trained",
      {"global", "maxsim", "maxsim_trained"} <= set(out))
check("no variance keys are produced (nothing to report, by design)",
      not {"global_var", "global_prob_mf"} & set(out))
check("the two MaxSims are genuinely different branches",
      not torch.allclose(out["maxsim"], out["maxsim_trained"], atol=1e-4))
check("maxsim (frozen) is computed on proj_g",
      torch.allclose(out["maxsim"],
                     symmetric_maxsim(head.project_g(ha), ma, head.project_g(hb), mb), atol=1e-5))
check("ORDER INVARIANT on every branch",
      all(torch.allclose(out[k], head(hb, mb, ha, ma)[k], atol=1e-5)
          for k in ("global", "maxsim", "maxsim_trained")))
pad = 6
o2 = head(torch.cat([ha, torch.randn(B, pad, D) * 40], 1),
          torch.cat([ma, torch.zeros(B, pad, dtype=torch.bool)], 1), hb, mb)
check("PADDING INVARIANT on every branch",
      all(torch.allclose(out[k], o2[k], atol=1e-3) for k in ("global", "maxsim", "maxsim_trained")))


# ---- A2. GRADIENT ISOLATION, re-checked with the uncertainty off ---------------------------------
def grads_from(loss_fn):
    head.zero_grad(set_to_none=True)
    h = ha.clone().requires_grad_(True)
    loss_fn(h).backward()
    got = lambda mod: any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    return {"encoder_input": h.grad is not None and h.grad.abs().sum() > 0,
            "proj_g": got(head.proj_g),
            "proj_m": head.proj_m.weight.grad is not None and head.proj_m.weight.grad.abs().sum() > 0,
            "global_head": got(head.g)}


def rank_like(h):
    xma, xmb = head.rank_inputs(h, hb)
    return symmetric_maxsim(xma, ma, xmb, mb).mean()


def global_like(h):
    return head(h, ma, hb, mb, with_trained_maxsim=False)["global"].mean()


g = grads_from(rank_like)
check("1. rank loss REACHES the encoder (its whole purpose)", g["encoder_input"])
check("2. rank loss does NOT touch proj_g", not g["proj_g"])
check("   rank loss does train proj_m", g["proj_m"])
check("   rank loss does NOT touch the Global head", not g["global_head"])
g = grads_from(global_like)
check("3. Global loss does NOT touch proj_m", not g["proj_m"])
check("   Global loss trains proj_g and the Global head", g["proj_g"] and g["global_head"])
check("   Global loss reaches the encoder", g["encoder_input"])

# ---- A3. the uncertainty plumbing is a safe no-op, not a crash -----------------------------------
try:
    head.reset_all_precisions()
    head.fit_all_covariances(reduce=False)
    head.accumulate_residue_precision(head.project_g(ha), ma)
    safe = True
except Exception as e:  # noqa: BLE001
    safe = False
    print(f"      raised: {e!r}")
check("reset/fit/accumulate are no-ops when there is no GP", safe)
rep = head.residue_report(head.project_g(ha), ma, head.project_g(hb), mb)
check("residue_report still gives the FREE diagnostics and no var_res",
      {"best", "sel_gap", "margin"} <= set(rep) and "var_res" not in rep)


# ---- B. the post-hoc add-on is fittable, and faithful to the logit -------------------------------
@torch.no_grad()
def penultimate(gb, pa, pb):
    """The Global head's features just before its final Linear, symmetrised the way the head symmetrises.
    gb.mlp = [Linear, ReLU, Dropout, Linear]; gb.mlp[:3] is everything up to the final Linear."""
    one = lambda u, v: gb.mlp[:3](gb.ln(torch.cat([u, v, (u - v).abs()], dim=-1)))
    return 0.5 * (one(pa, pb) + one(pb, pa))


from threebranch import masked_mean  # noqa: E402

pa, pb = masked_mean(head.project_g(ha), ma), masked_mean(head.project_g(hb), mb)
feat = penultimate(head.g, pa, pb)
check("penultimate features are recoverable at the head's hidden width",
      feat.shape == (B, head.g.mlp[0].out_features))
check("the model's OWN logit is exactly the final Linear of those features "
      "(so a post-hoc GP scores the same quantity)",
      torch.allclose(head.g.mlp[3](feat).squeeze(-1), out["global"], atol=1e-5))

gp = RFFGP(feat.shape[-1], rffs=64, likelihood="leverage", seed=5)
gp.calibrate_amplitude(feat, reduce=False)
gp.accumulate(gp.rff(feat))
gp.fit_covariance(reduce=False)
var = gp(feat, get_var=True)[1]
check("a leverage RFF-GP fits post hoc on them and returns a finite variance per pair",
      var.shape == (B,) and bool(torch.isfinite(var).all()) and bool((var > 0).all()))

print(f"\n{sum(OK)}/{len(OK)} checks pass")
sys.exit(0 if all(OK) else 1)
