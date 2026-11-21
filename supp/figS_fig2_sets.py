#!/usr/bin/env python3
"""Figure 2 v5 (2026-11-20): journal-style formatting of v4. Content as v4: a workflow | b AUPR all proteins | c AUPR no homolog | d verified direct contacts (human) | e AUPR versus AlphaFold-Multimer scores |
f precision of calls (with counts) | g keyword categories (all pairs and pairs new to the model) | h, i worked examples.
Formatting: 183 mm double-column canvas (1220 px; 12 px ~ 5 pt), NO panel titles and NO footnotes inside the figure (descriptions go in the legend), bold lowercase panel letters, one shared colour legend,
minimal in-panel labels, text >= 10.5 px (>= 12 px except dense value labels), roughly square. Run from this folder."""
import sys, numpy as np, pandas as pd

H = "<ROOT>/human_af_screen/bern_univ/"; F2 = "<ROOT>/human_af_screen/fig2/"
W, HT = 1220, 590; INK, INK2, GRID = "#111111", "#444444", "#e8e8e8"
COL = {"PanPPI": "#4A3FA0", "LoGoPPI": "#E0566A", "ProteomeLM": "#2E9E5B", "AFM ipSAE": "#3D4651", "AFM ipTM": "#9AA3AD", "AFM flag": "#B59B7A"}
FS_TICK, FS_AX, FS_LEG, FS_VAL = 12, 13, 12.5, 11
out = []; add = out.append
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, size=FS_TICK, weight="normal", anchor="start", fill=INK2, italic=False, rot=None):
    st = ' font-style="italic"' if italic else ""; tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{st}{tr}>{esc(s)}</text>')
def line(x1, y1, x2, y2, stroke=GRID, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""; add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>')
def circle(cx, cy, r, fill, stroke="none", sw=1.0): add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def letter(x, y, s): text(x, y, s, 20, "bold", fill=INK)
def yv(v, lo, hi, y0, y1): return y1 - (v - lo) / (hi - lo) * (y1 - y0)
def axes(x0, x1, y0, y1, lo, hi, ticks, label, fmt="%.1f"):
    for t in ticks:
        y = yv(t, lo, hi, y0, y1); line(x0, y, x1, y); text(x0 - 6, y + 4.2, fmt % t, FS_TICK, anchor="end")
    line(x0, y0, x0, y1, "#777777", 1.3); line(x0, y1, x1, y1, "#777777", 1.3); text(x0 - 44, (y0 + y1) / 2, label, FS_AX, anchor="middle", rot=-90, fill=INK)
def bar(cx, v, color, bw, lo, hi, y0, y1, dots=None, label=True, bold=False, size=FS_VAL, digits=2, op=1.0):
    top = yv(v, lo, hi, y0, y1); base = yv(lo, lo, hi, y0, y1); add(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(base - top, 0):.1f}" fill="{color}" fill-opacity="{op}"/>'); hp = top
    for s in (dots or []): ys = yv(s, lo, hi, y0, y1); circle(cx, ys, 2.5, "#ffffff", INK2, 1.2); hp = min(hp, ys)
    if label: text(cx, hp - 5, (("." + ("%0*d" % (digits, round(v * 10 ** digits)))) if v < 0.995 else "1.0"), size, "bold" if bold else "normal", anchor="middle", fill=INK)
NM = {"human": "Human", "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "yeast": "Yeast", "paer": "P. aer.", "mtb": "M. tb", "ecoli": "E. coli"}
R = pd.read_csv(H + "fig2_results_long.csv"); R = R[R.tier == "s700"]; CR = pd.read_csv(H + "cand_results.csv"); CP = pd.read_csv(H + "cand_call_precision.csv"); CM = pd.read_csv(H + "cand_match_afm.csv")
FOCUS = ["human", "mouse", "fly", "worm", "yeast", "ecoli"]   # focus species only: human plus the five D-SCRIPT test species (M. tuberculosis, P. aeruginosa excluded)
R = R[R.species.isin(FOCUS)]; CR = CR[CR.species.isin(FOCUS)]
for _d in (CP, CM):
    if "species" in _d.columns: _d.drop(_d[~_d.species.isin(FOCUS)].index, inplace=True)
CA = pd.read_csv(F2 + "contact_auroc.csv")
G0 = pd.read_csv(F2 + "func_enrich.csv").merge(pd.read_csv(F2 + "func_enrich_clean.csv"), on="keyword", how="left")
LX0, LX1, RX0, RX1 = 78, 585, 705, 1190; AXH = 145; TOPS = [100, 350, 585, 820]; GH = 262; U = "<ROOT>/human_af_screen/univ_nonhuman/"
"""Supplementary Figure (Fig. 2 sets): the proteome-scale degree-balanced evaluation behind the main Fig. 2. a, b AUPR per species for all proteins and for proteins with no training homolog (moved from Fig. 2 d, e);
c AUPR against AlphaFold-Multimer scores on the AlphaFold-DB candidate sets (all proteins light, no training homolog solid; STRING/PDB co-membership truth); d degree-only AUROC of every set (the control).
Built from human_af_screen/bern_univ/{fig2_results_long.csv,cand_results.csv}. Run from anywhere; writes supp/figS_fig2_sets.svg/.png."""
def aupr_panel(x0, x1, T, level, ltr, label):
    y0, y1 = T + 14, T + 14 + AXH; letter(6 if x0 < 300 else 633, T + 20, ltr)
    axes(x0, x1, y0, y1, 0, 1.0, [0, 0.5, 1.0], "AUPR"); D = R[R.level == level].set_index("species"); order = [s for s in ["human", "mouse", "fly", "worm", "yeast", "ecoli"] if s in D.index]; gw = (x1 - x0) / len(order); bw = gw / 3.45; flagged = []
    for i, spc in enumerate(order):
        cx = x0 + (i + 0.5) * gw; r = D.loc[spc]; fl = False; flagged.append((NM[spc], r.deg_only_auroc)) if fl else None
        for j, (k, c) in enumerate((("PanPPI blend aupr", COL["PanPPI"]), ("LoGoPPI blend aupr", COL["LoGoPPI"]), ("ProteomeLM aupr", COL["ProteomeLM"]))):
            bar(cx + (j - 1) * (bw + 1), r[k], c, bw, 0, 1.0, y0, y1, ([r["PanPPI blend aupr_min"], r["PanPPI blend aupr_max"]] if j == 0 else None), True, j == 0, 10.5, 2, 0.4 if fl else 1.0)
        text(cx, y1 + 16, NM[spc] + ("\u2020" if fl else ""), 12, anchor="middle", fill=INK, italic=(spc in ("paer", "mtb", "ecoli"))); text(cx, y1 + 30, "n=%s" % format(int(r.n_pos), ","), 10.5, anchor="middle")
        if spc == "human" and level == "L0": text(cx, y1 + 43, "training species", 10.5, anchor="middle", italic=True)
    text((x0 + x1) / 2, y1 + 64, "Species (%s)" % label, FS_AX, anchor="middle", fill=INK)
    if flagged: text((x0 + x1) / 2, y1 + 81, "\u2020 light bars: degree-only AUROC above our 0.60 limit (" + ", ".join("%s %.3f" % f for f in flagged) + ")", 11, anchor="middle")

# shared legend
ly = 18; x = 78
for nm in ("PanPPI", "LoGoPPI", "ProteomeLM", "AFM ipSAE", "AFM ipTM"):
    add(f'<rect x="{x}" y="{ly - 11}" width="13" height="13" rx="2" fill="{COL[nm]}"/>'); lab = "ProteomeLM-PPI" if nm == "ProteomeLM" else nm; text(x + 18, ly, lab, FS_LEG, "bold" if nm == "PanPPI" else "normal", fill=INK); x += 32 + 7.6 * len(lab)
aupr_panel(LX0, LX1, 40, "L0", "a", "all proteins"); aupr_panel(RX0, RX1, 40, "L2", "b", "only proteins with no homolog in the training set")
# c: AUPR against AFM scores
T = 300; y0, y1 = T + 14, T + 14 + AXH; letter(6, T + 20, "c"); axes(LX0, LX1, y0, y1, 0, 1.0, [0, 0.5, 1.0], "AUPR (median)")
C0 = CR[(CR.variant == "A") & CR.kept & (CR.level == "L0") & (CR.tier == "s700")]; C2 = CR[(CR.variant == "A") & CR.kept & (CR.level == "L2") & (CR.tier == "s700")]
items = [("PanPPI", "PanPPI blend aupr", COL["PanPPI"]), ("LoGoPPI", "LoGoPPI aupr", COL["LoGoPPI"]), ("ProteomeLM", "ProteomeLM aupr", COL["ProteomeLM"]), ("ipSAE", "AFM ipSAE aupr", COL["AFM ipSAE"]), ("ipTM", "AFM ipTM aupr", COL["AFM ipTM"])]; gwe = (LX1 - LX0) / len(items)
def bar2(cx, v, color, bw, dots, op, bold):
    top = yv(v, 0, 1.0, y0, y1); base = yv(0, 0, 1.0, y0, y1); add(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{base - top:.1f}" fill="{color}" fill-opacity="{op}"/>'); hp = top
    for d_ in dots: ys = yv(d_, 0, 1.0, y0, y1); circle(cx, ys, 2.3, "#ffffff", INK2, 1.1); hp = min(hp, ys)
    text(cx, hp - 5, "." + "%02d" % round(v * 100), 10.5, "bold" if bold else "normal", anchor="middle", fill=INK)
for g, (nm, k, c) in enumerate(items):
    cx = LX0 + (g + 0.5) * gwe; bar2(cx - 17, C0[k].median(), c, 31, C0[k].tolist(), 0.42, g == 0); bar2(cx + 17, C2[k].median(), c, 31, C2[k].tolist(), 1.0, g == 0); text(cx, y1 + 17, nm, 12, anchor="middle", fill=INK)
text((LX0 + LX1) / 2, y1 + 56, "AUPR against AlphaFold-Multimer scores on candidate sets", FS_AX, anchor="middle", fill=INK); text((LX0 + LX1) / 2, y1 + 73, "light: all proteins (%d sets); solid: no training homolog (%d sets)" % (len(C0), len(C2)), 12, anchor="middle")
# d: degree-only AUROC of every set
letter(633, T + 20, "d"); lo_, hi_ = 0.48, 0.66; axes(RX0, RX1, y0, y1, lo_, hi_, [0.5, 0.6], "Degree-only AUROC", "%.2f")
order = ["human", "mouse", "fly", "worm", "yeast", "ecoli"]; gwd = (RX1 - RX0) / len(order)
for lvl, filled in (("L0", True), ("L2", False)):
    D_ = R[R.level == lvl].set_index("species")
    for gi, spc in enumerate(order):
        if spc in D_.index:
            cx = RX0 + (gi + 0.5) * gwd + (-8 if lvl == "L0" else 8); v = float(D_.loc[spc].deg_only_auroc); yy = yv(v, lo_, hi_, y0, y1)
            circle(cx, yy, 5, INK2 if filled else "#ffffff", INK2, 1.6); text(cx, yy - 9, ("%.2f" % v).lstrip("0"), 10.5, anchor="middle", fill=INK)
for gi, spc in enumerate(order): text(RX0 + (gi + 0.5) * gwd, y1 + 17, NM[spc], 12, anchor="middle", fill=INK, italic=(spc in ("paer", "mtb", "ecoli")))
yy = yv(0.5, lo_, hi_, y0, y1); line(RX0, yy, RX1, yy, "#777777", 1.2, "4 3")
circle(RX0 + 40, y1 + 46, 5, INK2, INK2, 1.6); text(RX0 + 50, y1 + 50, "all proteins", 12, fill=INK); circle(RX0 + 160, y1 + 46, 5, "#ffffff", INK2, 1.6); text(RX0 + 170, y1 + 50, "no training homolog", 12, fill=INK)
text((RX0 + RX1) / 2, y1 + 73, "Degree-only AUROC of each evaluation set (0.5 = no degree signal)", FS_AX, anchor="middle", fill=INK)
body = "\n".join(out)
open("<ROOT>/supp/figS_fig2_sets.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}"><rect width="{W}" height="{HT}" fill="#ffffff"/>\n{body}\n</svg>'); print("wrote figS_fig2_sets.svg", W, "x", HT)
