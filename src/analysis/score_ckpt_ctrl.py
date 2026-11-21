"""Score saved CTRL checkpoints (epoch_XX.pt) with EXACTLY main_ctrl.py's final-report procedure, on one small GPU, without touching main_ctrl.py.
Imports main_ctrl as a module and reuses its own functions (Model, embed_cache, score_cached, fit_calibration, calibrate, auroc, aupr, read_pairs/read_fasta),
so the only thing this file adds is the loop over checkpoints.  Per checkpoint: calibration is refit on the human calibration split with that checkpoint's raw
scores (the same thing the original run did for best.pt), then the human selection split and every species are scored.  NO uncertainty (the CTRL model has none).
Usage (from the repo dir):  python -I analysis/score_ckpt_ctrl.py --ckpt runs_ft/<run>/output/checkpoints/epoch_05.pt,.../epoch_08.pt --out-dir DIR [--smoke]
"""
import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import main_ctrl as mc  # noqa: E402

EUK = ("yeast", "mouse", "fly", "worm")
PROK = ("ecoli", "208964", "224308", "83332")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--esm-dir", default="ext/esm2_650M_hf")
    p.add_argument("--data-dir", default="data/logoppi/data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=141)
    p.add_argument("--species", default=mc.FINAL_SPECIES)
    p.add_argument("--smoke", action="store_true", help="tiny subsets and yeast only: checks the pipeline end to end, numbers are NOT results")
    a = p.parse_args()
    dev = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731  (training used --precision bf16)
    D = os.path.abspath(a.data_dir)
    n_eval = 300 if a.smoke else None
    tr = mc.sample(mc.read_pairs(os.path.join(D, "human_train_clean.csv")), 2000 if a.smoke else None, a.seed)
    sel = mc.sample(mc.read_pairs(os.path.join(D, "human_validation_selection.csv")), n_eval, a.seed)
    cal = mc.sample(mc.read_pairs(os.path.join(D, "human_validation_calibration.csv")), n_eval, a.seed)
    seqs = mc.read_fasta(os.path.join(D, "proteins.fasta"))
    species = ["yeast"] if a.smoke else [s for s in a.species.split(",") if s]
    model = mc.Model(os.path.abspath(a.esm_dir), 1280, 800, grad_ckpt=False, use_latent=False, use_sn=False, use_gp=False, seed=a.seed, use_res_gp=False).to(dev)
    os.makedirs(a.out_dir, exist_ok=True)
    human = set(tr.a) | set(tr.b) | set(sel.a) | set(sel.b) | set(cal.a) | set(cal.b)
    for ck in a.ckpt.split(","):
        t0 = time.time()
        tag = os.path.splitext(os.path.basename(ck))[0]
        sd = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(sd)          # strict: a key mismatch must be an error, not a silent partial load
        del sd
        model.eval()
        print(f"===== {tag}: {ck}", flush=True)
        cache = mc.embed_cache(model, human, seqs, dev, amp)
        raw_tr = mc.score_cached(model, tr, cache, dev, batch=128)
        raw_cal = mc.score_cached(model, cal, cache, dev)
        raw_sel = mc.score_cached(model, sel, cache, dev)
        st = mc.fit_calibration(raw_tr["maxsim"], raw_cal, cal.label.values.astype(float))
        rows = []

        def report(name, df, raw):
            y = df.label.values.astype(bool)
            c = mc.calibrate(raw, st)
            line = f"  {name:16} n={len(y):6d} "
            for k in ("global", "maxsim", "logoppi_g+m"):
                rows.append(dict(ckpt=tag, set=name, model=k, auroc=mc.auroc(y, c[k]), aupr=mc.aupr(y, c[k]), n=len(y)))
                line += f"{k} {rows[-1]['aupr']:.4f}  "
            print(line, flush=True)

        report("human_selection", sel, raw_sel)
        del cache
        torch.cuda.empty_cache()
        for sp in species:
            pr = mc.sample(mc.read_pairs(os.path.join(D, "test", f"{sp}_pairs.csv")), n_eval, a.seed)
            sq = mc.read_fasta(os.path.join(D, "test", f"{sp}.fasta"))
            cache = mc.embed_cache(model, set(pr.a) | set(pr.b), sq, dev, amp)
            report(sp, pr, mc.score_cached(model, pr, cache, dev))
            del cache
            torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out_dir, f"{tag}{'_SMOKE' if a.smoke else ''}.csv"), index=False)
        b = df[df.model == "logoppi_g+m"].set_index("set")["aupr"]
        if not a.smoke and all(s in b.index for s in EUK + PROK):
            print(f"  MACRO g+m  eukaryote {np.mean([b[s] for s in EUK]):.4f}  prokaryote {np.mean([b[s] for s in PROK]):.4f}   ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
