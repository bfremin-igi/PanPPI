"""Fig 2 panel g data: which UniProt functional categories do PanPPI and the AlphaFold-Multimer flag recover? Human, STRING>=700 positives of the degree-matched candidate-negative cell (variant A, level L0).
PanPPI call = blend score above the 95th percentile of the negatives of each replicate (as panel f left), averaged over replicates; AFM call = confident flag. Category = UniProt keyword of either protein."""
import numpy as np, pandas as pd
H = "<ROOT>/human_af_screen/"; SEEDS = ("s42", "s7", "s141"); TIER = "s700"
ids = [l.strip() for l in open(H + "fig2/annot/human_ids.txt") if l.strip()]; B = pd.read_csv(H + "bern_univ/out3/human/cand_scores.tsv.gz", sep="\t"); B = B[(B.variant == "A") & (B.tier == TIER) & (B.level == "L0")]
B["pan"] = np.mean([0.5 * (B[f"g_{s}"] + B[f"m_{s}"]) for s in SEEDS], axis=0); P = B[B.label == 1].reset_index(drop=True); calls = np.zeros(len(P)); lgc = np.zeros(len(P)); reps = sorted(set(B[B.label == 0].rep))
for r in reps:
    N = B[(B.label == 0) & (B.rep == r)]; calls += (P.pan.to_numpy() > np.quantile(N.pan, .95)); lgc += (P.lg.to_numpy() > np.quantile(N.lg, .95))
P["pan_call"] = calls / len(reps); P["lg_call"] = lgc / len(reps); P["afm"] = P.afm_pass.astype(float)
K = pd.read_csv(H + "fig2/annot/uniprot_human_keywords.tsv", sep="\t"); kw = {r.Entry: set(str(r.Keywords).split(";")) for r in K.itertuples()}
BLOCK = {"Proteomics identification", "Reference proteome", "Alternative splicing", "Phosphoprotein", "Acetylation", "Direct protein sequencing", "Polymorphism", "Isopeptide bond", "Disulfide bond", "Glycoprotein", "Cytoplasm", "Methylation", "Lipoprotein", "Signal", "Transmembrane helix", "Transmembrane", "Membrane", "Repeat", "Coiled coil", "Disease variant", "3D-structure", "Complete proteome", "Citrullination", "Hydroxylation", "Palmitate", "Prenylation", "ADP-ribosylation", "Amidation", "Lipid-binding", "Metal-binding", "Zinc", "Nucleotide-binding", "ATP-binding", "Zinc-finger", "Cytoplasmic vesicle", "Ubl conjugation", "S-nitrosylation", "Pyrrolidone carboxylic acid"}
ent = [(a, b) for a, b in zip(P.i.map(lambda x: ids[x]), P.j.map(lambda x: ids[x]))]; sets = [kw.get(a, set()) | kw.get(b, set()) for a, b in ent]
cnt = pd.Series([k for s in sets for k in s]).value_counts(); cats = [k for k, v in cnt.items() if v >= 800 and k not in BLOCK]
rows = []
grp = P.i.to_numpy(); ug, gi = np.unique(grp, return_inverse=True); rng = np.random.default_rng(1)
for k in cats:
    m = np.array([k in s for s in sets]); n = int(m.sum())
    gs = np.bincount(gi[m], minlength=len(ug)); res = {"keyword": k, "n_pairs": n}
    for nm in ("afm", "pan_call", "lg_call"):
        res[nm] = P[nm][m].mean(); sm = np.bincount(gi[m], weights=P[nm].to_numpy()[m], minlength=len(ug)); bs = []
        for _ in range(300): w = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug)); bs.append((w * sm).sum() / max((w * gs).sum(), 1))
        res[nm + "_lo"], res[nm + "_hi"] = np.percentile(bs, [2.5, 97.5])
    rows.append(res)
T = pd.DataFrame(rows).sort_values("n_pairs", ascending=False); T.round(4).to_csv("func_enrich.csv", index=False)
print("all positives n=%d: AFM recall %.3f, PanPPI %.3f, LoGoPPI %.3f" % (len(P), P.afm.mean(), P.pan_call.mean(), P.lg_call.mean()))
pd.set_option("display.width", 220); print(T[["keyword", "n_pairs", "afm", "afm_lo", "afm_hi", "pan_call", "pan_call_lo", "pan_call_hi", "lg_call"]].round(3).to_string(index=False))
