"""CPU unit tests for the panppi package with a tiny stand-in encoder (no ESM weights, no transformers needed).
Run: python tests/test_panppi.py     (from the repository root, after `pip install -e .` or with the root on PYTHONPATH)"""
import json, os, sys, tempfile, types
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from panppi import model as M
from panppi import score as P
from panppi import weights as W

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name + (f"   [{extra}]" if extra else ""), flush=True)

D = 64
class StubTok:
    def __call__(self, seqs, return_tensors="pt", padding=True, truncation=True, max_length=802):
        L = max(len(s) for s in seqs) + 2; ids = torch.ones(len(seqs), L, dtype=torch.long); att = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            row = [0] + [4 + (ord(c) % 20) for c in s] + [2]; ids[i, :len(row)] = torch.tensor(row); att[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": att}
class StubEsm(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.emb = torch.nn.Embedding(32, D); self.lin = torch.nn.Linear(D, D); self.config = types.SimpleNamespace(position_embedding_type="absolute")
    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=torch.tanh(self.lin(self.emb(input_ids))))
def make(seed):
    torch.manual_seed(seed); return M.Model("stub", D, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=141, use_res_gp=False, encoder=StubEsm(), tok=StubTok())

tmp = tempfile.mkdtemp(); dev = torch.device("cpu")
# ---- input parsing
fa = os.path.join(tmp, "p.fasta"); open(fa, "w").write(">P1 some description\nMKV*\nLLA\n>P2\nACDEFGHIK\n")
s = P.read_fasta(fa); check("FASTA: id is the first word, '*' stripped, lines joined", s == {"P1": "MKVLLA", "P2": "ACDEFGHIK"})
open(fa, "w").write(">A\nMK\n>A\nMK\n")
try: P.read_fasta(fa); check("FASTA: duplicate ids raise", False)
except ValueError: check("FASTA: duplicate ids raise", True)
open(fa, "w").write(">A\n>B\nMK\n")
try: P.read_fasta(fa); check("FASTA: empty sequence raises", False)
except ValueError: check("FASTA: empty sequence raises", True)
pc = os.path.join(tmp, "pairs.csv"); pd.DataFrame({"query": ["P1"], "text": ["P2"], "extra": [7]}).to_csv(pc, index=False)
d = P.read_pairs(pc); check("pairs: query/text accepted and renamed, extra column kept", list(d.columns[:2]) == ["protein_a", "protein_b"] and "extra" in d.columns)
pd.DataFrame({"x": [1], "y": [2]}).to_csv(pc, index=False)
try: P.read_pairs(pc); check("pairs: unknown columns raise", False)
except ValueError: check("pairs: unknown columns raise", True)

# ---- model, checkpoint and calibration round trip
rs = np.random.RandomState(0); AA = "ACDEFGHIKLMNPQRSTVWY"
seqs = {f"P{i:02d}": "".join(rs.choice(list(AA), rs.randint(15, 70))) for i in range(14)}; ids = sorted(seqs)
model = make(1); wp = os.path.join(tmp, "m_best.pt"); torch.save(model.state_dict(), wp)
st = {"maxsim_train": {"mean": 0.3, "std": 0.2}, "global": {"temperature": 1.7, "bias": -0.4}, "maxsim": {"scale": 1.3, "bias": -0.2}}; cp = os.path.join(tmp, "m_calibration.json"); json.dump(st, open(cp, "w"))
m2, st2 = P.load_model(wp, cp, "stub", dev, encoder=StubEsm(), tok=StubTok(), d_enc=D)
check("weights load strictly and calibration is read", st2 == st)
bad = os.path.join(tmp, "bad.json"); json.dump({"global": {}}, open(bad, "w"))
try: P.load_model(wp, bad, "stub", dev, encoder=StubEsm(), tok=StubTok(), d_enc=D); check("calibration without maxsim raises", False)
except ValueError: check("calibration without maxsim raises", True)
cache = P.embed(m2, seqs, ids, dev, block=5)
check("every protein embedded once, one row per residue", all(cache[k].shape[0] == len(seqs[k]) for k in ids))
# ---- pairs mode equals the reference scorer
allp = pd.DataFrame([(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]], columns=["protein_a", "protein_b"])
g, m = P.score_pairs(m2, st, allp, cache, dev, batch=16, chunk=40)
raw = M.score_cached(m2, allp.rename(columns={"protein_a": "a", "protein_b": "b"}), cache, dev); c = M.calibrate(raw, st)
check("pairs mode = main_e3.score_cached + calibrate (chunked)", np.allclose(g, c["global"], atol=1e-5) and np.allclose(m, c["maxsim"], atol=1e-5))
g2, m2_ = P.score_pairs(m2, st, allp.rename(columns={"protein_a": "protein_b", "protein_b": "protein_a"}), cache, dev)
check("score is order-invariant (a,b) vs (b,a)", np.allclose(g, g2, atol=1e-4) and np.allclose(m, m2_, atol=1e-4), f"max {max(np.abs(g-g2).max(), np.abs(m-m2_).max()):.1e}")
# ---- all-vs-all equals brute force
av = P.AllVsAll(m2, st, cache, ids, dev, batch=4); P.self_check(av, m2, st, cache, dev, n=60); check("self-check passes on a random-weight model", True)
blend = 0.5 * (g + m); k = 20
I, J, G, Mx = av.top(k, progress_every=5)
want = set(zip(allp.protein_a[np.argsort(-blend)[:k]], allp.protein_b[np.argsort(-blend)[:k]])); got = set((ids[i], ids[j]) for i, j in zip(I, J))
check("all-vs-all top-k = brute-force top-k", want == got and len(I) == k)
check("all-vs-all result is sorted by blend, best first", np.all(np.diff(0.5 * (G + Mx)) <= 1e-9))
I2, J2, _, _ = P.AllVsAll(m2, st, cache, ids, dev, batch=3).top(k_top=91)
check("keeping every pair (k = N(N-1)/2) returns each pair once", len(set(zip(I2, J2))) == 91 and len(I2) == 91)
# ---- command line, both modes
load_model_real = P.load_model; P.load_model = lambda w, c, e, d: (m2, st)
fa2 = os.path.join(tmp, "q.fasta"); open(fa2, "w").write("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
pc2 = os.path.join(tmp, "pairs2.csv"); allp.head(30).rename(columns={"protein_a": "a", "protein_b": "b"}).to_csv(pc2, index=False); out1 = os.path.join(tmp, "o1.csv")
P.main(["--fasta", fa2, "--pairs", pc2, "--weights", wp, "--calibration", cp, "--out", out1, "--device", "cpu"])
o = pd.read_csv(out1); check("CLI pairs: columns and row count", list(o.columns) == ["protein_a", "protein_b", "global_logit", "maxsim_logit", "blend_logit", "probability"] and len(o) == 30)
check("CLI pairs: blend = mean of the two logits; probability = sigmoid", np.allclose(o.blend_logit, 0.5 * (o.global_logit + o.maxsim_logit), atol=2e-5) and np.allclose(o.probability, 1 / (1 + np.exp(-o.blend_logit)), atol=2e-5))
out2 = os.path.join(tmp, "o2.csv"); P.main(["--fasta", fa2, "--all-vs-all", "--top-fraction", "0.1", "--weights", wp, "--calibration", cp, "--out", out2, "--device", "cpu"])
o2 = pd.read_csv(out2); check("CLI all-vs-all: top 10% of 91 pairs = 10 rows, best first", len(o2) == 10 and np.all(np.diff(o2.blend_logit) <= 1e-9))
pd.DataFrame({"a": ["P00"], "b": ["NOPE"]}).to_csv(pc2, index=False)
try: P.main(["--fasta", fa2, "--pairs", pc2, "--weights", wp, "--calibration", cp, "--out", out1, "--device", "cpu"]); check("CLI: unknown protein in pairs exits with a message", False)
except SystemExit as e: check("CLI: unknown protein in pairs exits with a message", "not in the FASTA" in str(e))
try: P.main(["--fasta", fa2, "--all-vs-all", "--weights", wp, "--calibration", cp, "--out", out1, "--device", "cpu"]); check("CLI: all-vs-all without --top/--top-fraction exits", False)
except SystemExit as e: check("CLI: all-vs-all without --top/--top-fraction exits", "exactly one" in str(e))
# ---- model names and the weights cache
for bad_args in (dict(model="nope"),):
    try: W.resolve_weights(**bad_args); check("unknown model name raises a helpful error", False)
    except ValueError as e: check("unknown model name raises a helpful error", "Choose from" in str(e))
os.environ["PANPPI_CACHE"] = os.path.join(tmp, "cache"); W.WEIGHTS_BASE_URL = ""
try: W.resolve_weights("human"); check("missing weights without a download URL give instructions", False)
except FileNotFoundError as e: check("missing weights without a download URL give instructions", "README" in str(e))
os.makedirs(os.environ["PANPPI_CACHE"], exist_ok=True)
for n in ("weights_human_s42_best.pt", "weights_human_s42_calibration.json"): open(os.path.join(os.environ["PANPPI_CACHE"], n), "w").write("x")
w1, c1 = W.resolve_weights("human"); check("alias 'human' resolves to the cached seed-42 files", w1.endswith("weights_human_s42_best.pt") and c1.endswith("weights_human_s42_calibration.json"))
try: W.resolve_weights("human", weights=os.path.join(tmp, "missing.pt")); check("missing --weights file raises", False)
except FileNotFoundError: check("missing --weights file raises", True)
# ---- a failed download gives a helpful message, not a traceback
W.WEIGHTS_BASE_URL = "http://127.0.0.1:9/"; os.environ["PANPPI_CACHE"] = os.path.join(tmp, "cache2")
try: W.resolve_weights("bernett_s7"); check("failed download raises a message that names the record and --weights", False)
except RuntimeError as e: check("failed download raises a message that names the record and --weights", "zenodo.org/records/24407809" in str(e) and "--weights" in str(e))
check("no partial file is left behind after a failed download", not any(f.endswith(".part") for f in os.listdir(os.environ["PANPPI_CACHE"])))
# ---- the packaged code equals the training code (only if the original modules are next to this repository)
orig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tunaplus")
if os.path.isdir(orig):
    sys.path.insert(0, orig); import main_e3 as O
    mo = O.Model("stub", D, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=141, use_res_gp=False, encoder=StubEsm(), tok=StubTok()); torch.manual_seed(5)
    wp2 = os.path.join(tmp, "orig_best.pt"); torch.save(mo.state_dict(), wp2)
    mp, _ = load_model_real(wp2, cp, "stub", dev, encoder=StubEsm(), tok=StubTok(), d_enc=D)           # strict load of the ORIGINAL model's state dict into the package model
    prs = allp.head(40).rename(columns={"protein_a": "a", "protein_b": "b"})
    co = O.embed_cache(mo.eval(), ids, seqs, dev, lambda: P._NoAmp()); cp_ = P.embed(mp, seqs, ids, dev)
    ro = O.score_cached(mo, prs, co, dev); gp_, mp_ = P.score_pairs(mp, st, allp.head(40), cp_, dev)
    check("package scores = training-code scores on the same weights (strict load)", np.allclose(O.calibrate(ro, st)["global"], gp_, atol=1e-6) and np.allclose(O.calibrate(ro, st)["maxsim"], mp_, atol=1e-6))
else: print("SKIP  training-code comparison (tunaplus not next to this repository)")
print(f"\n{'ALL PASS' if ok else 'FAILURES'}")
sys.exit(0 if ok else 1)
