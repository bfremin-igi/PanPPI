"""panppi-score: score protein pairs with a trained PanPPI model.

Each protein is embedded ONCE, cached, and pairs are scored from the cache (global classifier + parameter-free MaxSim, calibrated; the reported
score is the mean of the two calibrated logits, "blend"). Two modes:

  pairs        score the pairs listed in a CSV   (--pairs pairs.csv)
  all-vs-all   score every pair among the FASTA proteins and keep the top ones   (--all-vs-all [--top N | --top-fraction F])

Examples
  panppi-score --fasta proteins.fasta --pairs pairs.csv --out scores.csv
  panppi-score --fasta proteome.fasta --all-vs-all --top-fraction 0.01 --model bernett --out top_pairs.csv

Input
  FASTA          protein id = first word of the header; sequences longer than 800 residues are truncated to the first 800 (as in training).
  pairs CSV      two columns naming the proteins, "a,b" (or "query,text" or "protein_a,protein_b"); extra columns are kept.
Output CSV       protein_a, protein_b, global_logit, maxsim_logit, blend_logit, probability  (probability = sigmoid(blend_logit)).
Models           --model human (default) or bernett downloads the checkpoint once; or give --weights file.pt (state dict) with --calibration file.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import model as M  # noqa: E402  (Model, embed_cache, score_cached, calibrate: the code that trained and evaluated the manuscript models)
from .weights import resolve_weights  # noqa: E402

MAX_LEN = 800
D_ENC = 1280
DEFAULT_ESM = "facebook/esm2_t33_650M_UR50D"


# ----------------------------------------------------------------------------------------------------------- input
def read_fasta(path):
    seqs, k = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                k = line[1:].split()[0] if line[1:].split() else ""
                if not k:
                    raise ValueError("FASTA record with an empty identifier")
                if k in seqs:
                    raise ValueError(f"duplicate FASTA identifier: {k}")
                seqs[k] = ""
            elif line:
                if k is None:
                    raise ValueError("FASTA: sequence before the first header")
                seqs[k] += line.replace("*", "").replace(" ", "").upper()
    empty = [k for k, s in seqs.items() if not s]
    if empty:
        raise ValueError(f"empty sequences: {empty[:5]}")
    return seqs


def read_pairs(path):
    d = pd.read_csv(path, dtype=str)
    for a, b in (("a", "b"), ("query", "text"), ("protein_a", "protein_b")):
        if a in d.columns and b in d.columns:
            d = d.rename(columns={a: "protein_a", b: "protein_b"})
            return d
    raise ValueError(f"{path}: need columns a,b (or query,text or protein_a,protein_b); found {list(d.columns)}")


def load_model(weights, calibration, esm, device, encoder=None, tok=None, d_enc=D_ENC):
    """Model with the trained PanPPI (E3a) configuration; strict state-dict load so a mismatch is an error. (encoder/tok/d_enc are for tests.)"""
    try:                                    # the encoder loads without its language-model head; that notice is expected and only confuses users
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    model = M.Model(esm, d_enc, MAX_LEN, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=141, use_res_gp=False, encoder=encoder, tok=tok).to(device)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    st = json.load(open(calibration))
    for key in ("global", "maxsim", "maxsim_train"):
        if key not in st:
            raise ValueError(f"{calibration}: missing '{key}' (not a PanPPI calibration file)")
    return model, st


class _NoAmp:
    def __enter__(self):
        return None

    def __exit__(self, *x):
        return False


def embed(model, seqs, ids, device, bf16=False, block=4000):
    """Embed each protein once (length-sorted batches); cache stays on the CPU in fp16, as in the manuscript."""
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if bf16 else (lambda: _NoAmp())
    cache = {}
    ids = list(ids)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ids), block):
            blk = M.embed_cache(model, ids[i:i + block], seqs, device, amp)
            for k, v in blk.items():
                cache[k] = v.cpu()
            print(f"  embedded {min(i + block, len(ids)):,}/{len(ids):,} proteins ({time.time() - t0:.0f} s)", flush=True)
    return cache


# ----------------------------------------------------------------------------------------------------------- pairs mode
def score_pairs(model, st, pairs, cache, device, batch=128, chunk=200_000):
    """Calibrated global, maxsim and blend logits for a pairs frame with columns protein_a, protein_b."""
    P = pairs.rename(columns={"protein_a": "a", "protein_b": "b"})[["a", "b"]]
    out = {"global": [], "maxsim": []}
    for i in range(0, len(P), chunk):
        raw = M.score_cached(model, P.iloc[i:i + chunk], cache, device, batch=batch)
        c = M.calibrate(raw, st)
        out["global"].append(c["global"])
        out["maxsim"].append(c["maxsim"])
    g, m = np.concatenate(out["global"]), np.concatenate(out["maxsim"])
    return g, m


# ----------------------------------------------------------------------------------------------------------- all-vs-all mode
class AllVsAll:
    """One anchor protein against many partners per batched matmul (same maths as main_e3.score_cached + calibrate; checked against it)."""

    def __init__(self, model, st, cache, ids, device, batch=256):
        self.model, self.ids, self.dev, self.batch = model, list(ids), device, batch
        self.N = len(self.ids)
        self.lens = np.array([cache[k].shape[0] for k in self.ids])
        self.off = np.r_[0, np.cumsum(self.lens)]
        self.X = torch.cat([cache[k] for k in self.ids]).to(device)
        self.offt = torch.tensor(self.off[:-1], device=device)
        self.lent = torch.tensor(self.lens, device=device)
        self.pooled = torch.stack([self.X[self.off[i]:self.off[i + 1]].float().mean(0) for i in range(self.N)])
        self.gT, self.gb = st["global"]["temperature"], st["global"]["bias"]
        self.mmu, self.msd = st["maxsim_train"]["mean"], st["maxsim_train"]["std"]
        self.msc, self.mb = st["maxsim"]["scale"], st["maxsim"]["bias"]
        self.neg = torch.finfo(torch.float32).min

    @torch.no_grad()
    def anchor(self, i, partners):
        """(calibrated global, calibrated maxsim) of pairs (i, partners); partners is an int array."""
        xa = self.X[self.off[i]:self.off[i + 1]].float()
        pa = self.pooled[i]
        order = np.argsort(self.lens[partners], kind="stable")
        ps = partners[order]
        og, om = np.empty(len(ps), np.float32), np.empty(len(ps), np.float32)
        for c in range(0, len(ps), self.batch):
            idx = torch.tensor(ps[c:c + self.batch], device=self.dev)
            L = self.lent[idx]
            Lmax = int(L.max())
            ar = torch.arange(Lmax, device=self.dev)
            gi = (self.offt[idx][:, None] + ar[None]).clamp_max(self.X.shape[0] - 1)
            mask = ar[None] < L[:, None]
            xb = self.X[gi].float()
            g = self.model.head.g(pa[None].expand(len(idx), -1), self.pooled[idx])
            sim = torch.matmul(xa[None], xb.transpose(1, 2))
            best_a = sim.masked_fill(~mask[:, None, :], self.neg).max(dim=2).values
            best_b = sim.max(dim=1).values
            m = 0.5 * (best_a.mean(1) + (best_b * mask).sum(1) / L.clamp_min(1))
            og[c:c + len(idx)] = (g / self.gT + self.gb).cpu().numpy()
            om[c:c + len(idx)] = (self.msc * (m - self.mmu) / self.msd + self.mb).cpu().numpy()
        g_out, m_out = np.empty_like(og), np.empty_like(om)
        g_out[order], m_out[order] = og, om
        return g_out, m_out

    def top(self, k_top, progress_every=200):
        """Top k_top pairs by blend logit over all i < j. Returns arrays (i, j, global, maxsim), best first."""
        pool, pool_n = [], 0
        N, NT, t0, done = self.N, self.N * (self.N - 1) // 2, time.time(), 0

        def prune():
            nonlocal pool, pool_n
            cat = [np.concatenate([p[c] for p in pool]) for c in range(5)]          # i, j, global, maxsim, blend
            if len(cat[4]) > k_top:
                keep = np.argpartition(-cat[4], k_top - 1)[:k_top]
                cat = [x[keep] for x in cat]
            pool, pool_n = [cat], len(cat[4])

        for i in range(N - 1):
            partners = np.arange(i + 1, N)
            g, m = self.anchor(i, partners)
            b = 0.5 * (g + m)
            sel = np.argpartition(-b, k_top - 1)[:k_top] if len(b) > k_top else np.arange(len(b))
            pool.append((np.full(len(sel), i), partners[sel], g[sel], m[sel], b[sel]))
            pool_n += len(sel)
            if pool_n > 2 * k_top:
                prune()
            done += len(partners)
            if (i + 1) % progress_every == 0 or i == N - 2:
                rate = done / (time.time() - t0)
                print(f"  anchor {i + 1:,}/{N:,}: {done:,}/{NT:,} pairs, {rate:,.0f} pairs/s, ETA {(NT - done) / max(rate, 1) / 60:.1f} min", flush=True)
        prune()
        I, J, G, Mx, B = pool[0]
        o = np.argsort(-B, kind="stable")
        return I[o], J[o], G[o], Mx[o]


def self_check(av, model, st, cache, device, n=200, seed=1):
    """Anchor kernel against the unchanged scorer on random pairs. Raises if the difference exceeds 5e-3 logits."""
    rs = np.random.RandomState(seed)
    ii, jj = rs.randint(0, av.N, n), rs.randint(0, av.N, n)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    P = pd.DataFrame({"protein_a": [av.ids[k] for k in ii], "protein_b": [av.ids[k] for k in jj]})
    need = set(P.protein_a) | set(P.protein_b)
    g_ref, m_ref = score_pairs(model, st, P, {k: cache[k].to(device) for k in need}, device)
    g_new = np.empty(len(ii)); m_new = np.empty(len(ii))
    for k in range(len(ii)):
        g, m = av.anchor(int(ii[k]), np.array([jj[k]]))
        g_new[k], m_new[k] = g[0], m[0]
    dg, dm = float(np.abs(g_new - g_ref).max()), float(np.abs(m_new - m_ref).max())
    print(f"  self-check on {len(ii)} random pairs: max |global diff| {dg:.1e}, max |maxsim diff| {dm:.1e}", flush=True)
    if dg > 5e-3 or dm > 5e-3:
        raise RuntimeError("all-vs-all kernel disagrees with the reference scorer; aborting")


# ----------------------------------------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(description="Score protein pairs with a trained PanPPI model.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--fasta", required=True, help="proteins (id = first word of the header)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pairs", help="CSV of pairs to score (columns a,b)")
    mode.add_argument("--all-vs-all", action="store_true", help="score every pair among the FASTA proteins and keep the top ones")
    p.add_argument("--model", default="human", help="human (default), bernett, or a specific run such as human_s7 / bernett_s141; downloaded on first use")
    p.add_argument("--weights", help="use your own checkpoint file instead of --model")
    p.add_argument("--calibration", help="calibration JSON for --weights (default: next to the weights)")
    p.add_argument("--esm", default=DEFAULT_ESM, help=f"ESM-2 650M Hugging Face id or local folder (default {DEFAULT_ESM})")
    p.add_argument("--out", required=True, help="output CSV")
    p.add_argument("--top", type=int, help="all-vs-all: keep the top N pairs")
    p.add_argument("--top-fraction", type=float, help="all-vs-all: keep the top fraction of all pairs (e.g. 0.01)")
    p.add_argument("--device", default="auto", help="auto, cuda or cpu")
    p.add_argument("--bf16", action="store_true", help="embed in bfloat16 (faster; what the manuscript's benchmark scores used; default is 32-bit)")
    p.add_argument("--batch", type=int, default=256, help="partners per matrix multiplication")
    p.add_argument("--no-check", action="store_true", help="skip the all-vs-all self-check")
    p.add_argument("--version", action="version", version="panppi " + __import__("panppi").__version__)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    if device.type == "cpu":
        print("WARNING: running on CPU; embedding the 650M encoder is slow (hours for a proteome). Use a GPU for real work.", flush=True)
    # check the inputs first, so a typo does not cost a multi-GB model download
    try:
        seqs = read_fasta(a.fasta)
        pairs = read_pairs(a.pairs) if a.pairs else None
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"input error: {e}")
    long_ = [k for k, s in seqs.items() if len(s) > MAX_LEN]
    if long_:
        print(f"note: {len(long_)} proteins are longer than {MAX_LEN} residues; only the first {MAX_LEN} are used (as in training)", flush=True)
    if a.pairs:
        missing = sorted((set(pairs.protein_a) | set(pairs.protein_b)) - set(seqs))
        if missing:
            sys.exit(f"{len(missing)} proteins in the pairs file are not in the FASTA, e.g. {missing[:5]}")
        ids = sorted(set(pairs.protein_a) | set(pairs.protein_b))
    else:
        if (a.top is None) == (a.top_fraction is None):
            sys.exit("all-vs-all needs exactly one of --top or --top-fraction")
        ids = sorted(seqs)
        if len(ids) < 2:
            sys.exit("need at least two proteins")
    try:
        weights, calib = resolve_weights(a.model, a.weights, a.calibration)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(str(e))
    t0 = time.time()
    print(f"loading model on {device} ...", flush=True)
    model, st = load_model(weights, calib, a.esm, device)
    print(f"embedding {len(ids):,} proteins once ...", flush=True)
    cache = embed(model, seqs, ids, device, bf16=a.bf16)
    if a.pairs:
        dev_cache = {k: v.to(device) for k, v in cache.items()} if device.type == "cuda" else cache
        g, m = score_pairs(model, st, pairs, dev_cache, device)
        out = pairs.copy()
        out["global_logit"], out["maxsim_logit"] = g, m
    else:
        N = len(ids)
        NT = N * (N - 1) // 2
        k_top = a.top if a.top is not None else max(1, int(math.ceil(a.top_fraction * NT)))
        k_top = min(k_top, NT)
        print(f"scoring {NT:,} pairs, keeping the top {k_top:,}", flush=True)
        av = AllVsAll(model, st, cache, ids, device, batch=a.batch)
        if not a.no_check:
            self_check(av, model, st, cache, device)
        I, J, g, m = av.top(k_top)
        out = pd.DataFrame({"protein_a": [ids[i] for i in I], "protein_b": [ids[j] for j in J], "global_logit": g, "maxsim_logit": m})
    out["blend_logit"] = 0.5 * (out["global_logit"] + out["maxsim_logit"])
    out["probability"] = 1.0 / (1.0 + np.exp(-out["blend_logit"].clip(-60, 60)))
    out.to_csv(a.out, index=False, float_format="%.5f")
    print(f"wrote {a.out}: {len(out):,} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


