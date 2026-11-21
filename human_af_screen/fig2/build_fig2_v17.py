#!/usr/bin/env python3
"""Figure 2 v5 (2026-11-20): journal-style formatting of v4. Content as v4: a workflow | b AUPR all proteins | c AUPR no homolog | d verified direct contacts (human) | e AUPR versus AlphaFold-Multimer scores |
f precision of calls (with counts) | g keyword categories (all pairs and pairs new to the model) | h, i worked examples.
Formatting: 183 mm double-column canvas (1220 px; 12 px ~ 5 pt), NO panel titles and NO footnotes inside the figure (descriptions go in the legend), bold lowercase panel letters, one shared colour legend,
minimal in-panel labels, text >= 10.5 px (>= 12 px except dense value labels), roughly square. Run from this folder."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "<ROOT>/human_af_screen/fig2"); import struct_panel as sp
H = "<ROOT>/human_af_screen/bern_univ/"; F2 = "<ROOT>/human_af_screen/fig2/"
W, HT = 1220, 905; INK, INK2, GRID = "#111111", "#444444", "#e8e8e8"
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
CA = pd.read_csv(F2 + "contact_auroc.csv")
G0 = pd.read_csv(F2 + "func_enrich.csv").merge(pd.read_csv(F2 + "func_enrich_clean.csv"), on="keyword", how="left")
LX0, LX1, RX0, RX1 = 78, 585, 705, 1190; AXH = 145; TOPS = [100, 350, 585]; GH = 262; U = "<ROOT>/human_af_screen/univ_nonhuman/"
# ---------------------------------------------------------------- a: workflow
letter(6, 24, "a"); ay = 8
def box(x, w, title, sub, fill, tc=INK):
    add(f'<rect x="{x}" y="{ay}" width="{w}" height="54" rx="7" fill="{fill}" stroke="#bbbbbb"/>'); text(x + w / 2, ay + 22, title, 14, "bold", anchor="middle", fill=tc); text(x + w / 2, ay + 42, sub, 12, anchor="middle", fill=tc if tc != INK else INK2)
def arrow(x1, x2): line(x1, ay + 27, x2, ay + 27, "#777777", 2); add(f'<polygon points="{x2},{ay + 27} {x2 - 9},{ay + 22} {x2 - 9},{ay + 32}" fill="#777777"/>')
box(40, 270, "All pairs in a proteome", "10\u2077 to 10\u2078 candidate pairs", "#f3f3f3"); arrow(312, 362); box(364, 280, "PanPPI", "~0.75 million pairs per minute on one H200 GPU", COL["PanPPI"], "#ffffff"); arrow(646, 696)
box(698, 240, "Top 1% of pairs", "keeps 52–88% of AFM hits", "#f3f3f3"); arrow(940, 990); box(992, 210, "AlphaFold-Multimer", "52–88\u00d7 fewer runs per hit", "#eceef0")
# shared legend
ly = 86; _names = ("PanPPI", "LoGoPPI", "ProteomeLM-PPI", "AFM ipSAE", "AFM ipTM", "AFM high-confidence call"); _tot = sum(32 + 7.6 * len(n) for n in _names) + 18 + 6.4 * len("PanPPI on pairs new to the model (e)"); x = (W - _tot) / 2
for nm in ("PanPPI", "LoGoPPI", "ProteomeLM", "AFM ipSAE", "AFM ipTM", "AFM flag"):
    add(f'<rect x="{x}" y="{ly - 11}" width="13" height="13" rx="2" fill="{COL[nm]}"/>'); lab = {"ProteomeLM": "ProteomeLM-PPI", "AFM flag": "AFM high-confidence call"}.get(nm, nm); text(x + 18, ly, lab, FS_LEG, "bold" if nm == "PanPPI" else "normal", fill=INK); x += 32 + 7.6 * len(lab)
circle(x + 8, ly - 4, 5, "none", COL["PanPPI"], 1.8); text(x + 18, ly, "PanPPI on pairs new to the model (e)", FS_LEG, fill=INK)
# ---------------------------------------------------------------- b, c: AUPR per species
def aupr_panel(x0, x1, T, level, ltr, label):
    y0, y1 = T + 14, T + 14 + AXH; letter(6 if x0 < 300 else 633, T + 20, ltr)
    axes(x0, x1, y0, y1, 0, 1.0, [0, 0.5, 1.0], "AUPR"); D = R[R.level == level].set_index("species"); order = [s for s in ["human", "mouse", "fly", "worm", "yeast", "paer", "mtb", "ecoli"] if s in D.index]; gw = (x1 - x0) / len(order); bw = gw / 3.45; flagged = []
    for i, spc in enumerate(order):
        cx = x0 + (i + 0.5) * gw; r = D.loc[spc]; fl = False; flagged.append((NM[spc], r.deg_only_auroc)) if fl else None
        for j, (k, c) in enumerate((("PanPPI blend aupr", COL["PanPPI"]), ("LoGoPPI blend aupr", COL["LoGoPPI"]), ("ProteomeLM aupr", COL["ProteomeLM"]))):
            bar(cx + (j - 1) * (bw + 1), r[k], c, bw, 0, 1.0, y0, y1, ([r["PanPPI blend aupr_min"], r["PanPPI blend aupr_max"]] if j == 0 else None), True, j == 0, 10.5, 2, 0.4 if fl else 1.0)
        text(cx, y1 + 16, NM[spc] + ("\u2020" if fl else ""), 12, anchor="middle", fill=INK, italic=(spc in ("paer", "mtb", "ecoli"))); text(cx, y1 + 30, "n=%s" % format(int(r.n_pos), ","), 10.5, anchor="middle")
        if spc == "human" and level == "L0": text(cx, y1 + 43, "training species", 10.5, anchor="middle", italic=True)
    text((x0 + x1) / 2, y1 + 64, "Species (%s)" % label, FS_AX, anchor="middle", fill=INK)
    if flagged: text((x0 + x1) / 2, y1 + 81, "\u2020 light bars: degree-only AUROC above our 0.60 limit (" + ", ".join("%s %.3f" % f for f in flagged) + ")", 11, anchor="middle")
# (AUPR panels d, e moved to the supplement: supp/figS_fig2_sets.py)
# ---------------------------------------------------------------- d: verified contacts and same-complex pairs without contact, by training overlap (one panel, four groups)
T = TOPS[1]; y0, y1 = T + 14, T + 14 + AXH; letter(6, T + 20, "d"); CD = pd.read_csv(F2 + "contact_direct_vs_indirect.csv"); XL, XR = LX0, RX1
axes(XL, XR, y0, y1, 0.5, 1.0, [0.5, 0.75, 1.0], "AUPR (degree-matched)", "%.2f"); line(XL, y1, XR, y1, "#777777", 1.3)
ctools = [("PanPPI", COL["PanPPI"]), ("LoGoPPI", COL["LoGoPPI"]), ("ProteomeLM", COL["ProteomeLM"]), ("AFM ipSAE", COL["AFM ipSAE"]), ("AFM ipTM", COL["AFM ipTM"])]; gwd = (XR - XL) / 4
labs = ["pair or homologous pair in training", "new pair (not in training, no homologous pair)", "pair or homologous pair in training", "new pair (not in training, no homologous pair)"]
for g in range(4):
    cx = XL + (g + 0.5) * gwd; r = CD.iloc[g]
    for k, (nm, c) in enumerate(ctools):
        bx_ = cx + (k - 2) * 47; bar(bx_, float(r[nm + " aupr"]), c, 43, 0.5, 1.0, y0, y1, None, False)
        ylo, yhi = yv(float(r[nm + " aupr_lo"]), 0.5, 1.0, y0, y1), yv(float(r[nm + " aupr_hi"]), 0.5, 1.0, y0, y1); line(bx_, ylo, bx_, yhi, INK, 1.4); line(bx_ - 4, ylo, bx_ + 4, ylo, INK, 1.4); line(bx_ - 4, yhi, bx_ + 4, yhi, INK, 1.4)
        text(bx_, yhi - 5, ("%.2f" % float(r[nm + " aupr"])).lstrip("0"), FS_VAL, "bold" if nm == "PanPPI" else "normal", anchor="middle", fill=INK)
    text(cx, y1 + 18, labs[g], 12, anchor="middle", fill=INK); text(cx, y1 + 33, "n=%s pairs" % format(int(r.n_pos), ","), 11, anchor="middle")
for a_, b_, lab in ((0, 1, "Direct contact (verified interface)"), (2, 3, "Same PDB entry, no direct contact")):
    xa_, xb_ = XL + (a_ + 0.08) * gwd, XL + (b_ + 0.92) * gwd; yb_ = y1 + 50; line(xa_, yb_, xb_, yb_, "#777777", 1.2); text((xa_ + xb_) / 2, yb_ + 18, lab, FS_AX, anchor="middle", fill=INK)
line(XL + 2 * gwd, y0, XL + 2 * gwd, y1 + 42, "#cccccc", 1, "4 3")
# ---------------------------------------------------------------- b: retention of AFM-confident interactions in the shortlist
T = TOPS[0]; letter(6, T + 20, "b")
RC = pd.read_csv(F2 + "retention_ci.csv"); RX = pd.read_csv(F2 + "retention_extended.csv"); from types import SimpleNamespace as NS; XL = {"human (clean)": "Human (new pairs)", "yeast": "Yeast", "mouse": "Mouse", "fly": "Fly", "worm": "Worm", "ecoli": "E. coli"}
RT = pd.concat([pd.read_csv(U + "recall_table.csv"), pd.read_csv(U + "unmatched_recall_mouse_worm_and_others.csv").rename(columns={"species": "set"})]); RT = RT[RT.truth == "AFM-confident"].drop_duplicates(["set", "tool"])
sets_ = [("human (clean)", "Human, new pairs"), ("yeast", "Yeast"), ("mouse", "Mouse"), ("fly", "Fly"), ("worm", "Worm"), ("ecoli", "E. coli")]; RH = 26; ya = T + 44
for fi, (col, ttl, fx0, fx1, frac) in enumerate((("@0.0001", "Top 0.01%", 128, 232, 1e-4), ("@0.001", "Top 0.1%", 258, 362, 0.001), ("@0.01", "Top 1%", 388, 492, 0.01), ("@0.1", "Top 10%", 518, 622, 0.10))):
    text((fx0 + fx1) / 2, T + 24, ttl, 12.5, "bold", anchor="middle", fill=INK); yb = ya + 10 + RH * (len(sets_) - 1) + 14
    for k, (sk, lab) in enumerate(sets_):
        y = ya + 10 + RH * k
        if fi == 0: text(fx0 - 12, y + 4, lab, 12, anchor="end", fill=INK, italic=(sk == "ecoli"))
        line(fx0, y, fx1, y, "#e3e3e3", 1.2); pr = RT[(RT.set == sk) & (RT.tool == "PanPPI blend")].iloc[0]; lg = RT[(RT.set == sk) & (RT.tool == "LoGoPPI")].iloc[0]
        _r = RX[(RX.set == XL[sk]) & np.isclose(RX.fraction, frac)].iloc[0]; cr = NS(pan=_r["PanPPI blend"], pan_lo=_r["PanPPI blend lo"], pan_hi=_r["PanPPI blend hi"], lg=_r["LoGoPPI"], lg_lo=_r["LoGoPPI lo"], lg_hi=_r["LoGoPPI hi"]); yp, yl = y - 4.5, y + 4.5
        for (lo_, hi_, yy_, c_) in ((cr.pan_lo, cr.pan_hi, yp, COL["PanPPI"]), (cr.lg_lo, cr.lg_hi, yl, COL["LoGoPPI"])): line(fx0 + lo_ * (fx1 - fx0), yy_, fx0 + hi_ * (fx1 - fx0), yy_, c_, 2.2); line(fx0 + lo_ * (fx1 - fx0), yy_ - 2.5, fx0 + lo_ * (fx1 - fx0), yy_ + 2.5, c_, 1.6); line(fx0 + hi_ * (fx1 - fx0), yy_ - 2.5, fx0 + hi_ * (fx1 - fx0), yy_ + 2.5, c_, 1.6)
        xx = fx0 + cr.pan * (fx1 - fx0); circle(xx, yp, 4.4, COL["PanPPI"], "#ffffff", 1.2); circle(fx0 + cr.lg * (fx1 - fx0), yl, 3.8, COL["LoGoPPI"], "#ffffff", 1.1); text(xx + (9 if cr.pan < 0.08 else 0), yp - 8, "%.0f%%" % (100 * cr.pan), 10.5, "bold", anchor="start" if cr.pan < 0.08 else "middle", fill=INK)
    xr = fx0 + frac * (fx1 - fx0); line(xr, ya + 2, xr, yb, "#999999", 1.2, "4 3")
    for t in (0, 0.5, 1.0): xx = fx0 + t * (fx1 - fx0); line(xx, yb, xx, yb + 4, "#777777", 1.2); text(xx, yb + 17, "%.1f" % t, FS_TICK, anchor="middle")
text((128 + 622) / 2, yb + 38, "Fraction of AFM-confident pairs kept in the top-ranked pairs", FS_AX, anchor="middle", fill=INK)
text((128 + 622) / 2, yb + 53, "1% of all pairs = 1.9M (human), 2.1M (mouse), 1.9M (worm), 0.8M (fly), 0.18M (yeast), 0.08M (E. coli)", 10.5, anchor="middle", fill=INK2)
# ---------------------------------------------------------------- c: precision against coverage (true pairs found), AFM flag vs PanPPI vs LoGoPPI at two operating points
T = TOPS[0]; letter(633, T + 20, "c"); PC = pd.read_csv(F2 + "precision_ci.csv").set_index("item")
y0f, y1f = T + 30, T + 14 + AXH + 16; plo, phi = 0.65, 1.0; axes(RX0, RX1, y0f, y1f, plo, phi, [0.7, 0.8, 0.9, 1.0], "Precision of calls", "%.1f")
xl0, xl1 = np.log(3000), np.log(60000); X = lambda v: RX0 + (np.log(v) - xl0) / (xl1 - xl0) * (RX1 - RX0)
for v, lab in ((5000, "5k"), (10000, "10k"), (20000, "20k"), (40000, "40k")): line(X(v), y0f, X(v), y1f, GRID); text(X(v), y1f + 17, lab, FS_TICK, anchor="middle")
text((RX0 + RX1) / 2, y1f + 40, "True pairs found (log scale, per replicate)", FS_AX, anchor="middle", fill=INK)
def pt(item, col):
    row = PC.loc[item]; x = X(row.true / 3); yy = yv(row.precision, plo, phi, y0f, y1f); lo, hi = yv(row.lo, plo, phi, y0f, y1f), yv(row.hi, plo, phi, y0f, y1f)
    line(x, lo, x, hi, col, 2.0); return x, yy
xa, ya_ = pt("afm", COL["AFM flag"]); circle(xa, ya_, 9, COL["AFM flag"], "#ffffff", 1.4)
for tool, c, rr in (("lg", COL["LoGoPPI"], 5.5), ("pm", COL["ProteomeLM"], 5.5), ("pan", COL["PanPPI"], 6.5)):
    xm, ym = pt(tool + "_match", c); xt, yt = pt(tool + "_top5", c); line(xm, ym, xt, yt, c, 2.0, "5 3"); circle(xt, yt, rr + 1, c, "#ffffff", 1.4); circle(xm, ym, rr - 1.5 if tool == "pan" else rr, c, "#ffffff", 1.3)
text(xa + 14, ya_ - 9, "AFM high-confidence call", 12, anchor="start", fill=INK); text(X(PC.loc["pan_top5"].true / 3) - 4, yv(PC.loc["pan_top5"].precision, plo, phi, y0f, y1f) - 20, "PanPPI: %.1f\u00d7 the true pairs at %.2f" % (PC.loc["pan_top5"].true / PC.loc["afm"].true, PC.loc["pan_top5"].precision), 12, "bold", anchor="end", fill=INK)
_xl, _yl = X(PC.loc["lg_top5"].true / 3), yv(PC.loc["lg_top5"].precision, plo, phi, y0f, y1f); _xp, _yp = X(PC.loc["pm_top5"].true / 3), yv(PC.loc["pm_top5"].precision, plo, phi, y0f, y1f)
text(_xl + 11, _yl - 3, "LoGoPPI: %.1f\u00d7 at %.2f" % (PC.loc["lg_top5"].true / PC.loc["afm"].true, PC.loc["lg_top5"].precision), 12, anchor="start", fill=INK); text(_xp + 11, _yp + 15, "ProteomeLM-PPI: %.1f\u00d7 at %.2f" % (PC.loc["pm_top5"].true / PC.loc["afm"].true, PC.loc["pm_top5"].precision), 12, anchor="start", fill=INK)
text((RX0 + RX1) / 2, y1f + 58, "left end of each line: same number of calls as AFM; right end: top 5% of negatives", 11.5, anchor="middle", fill=INK2)
# ---------------------------------------------------------------- g: keyword categories
T = TOPS[2]; letter(6, T + 20, "e")
DIS = {"Intellectual disability", "Primary mitochondrial disease", "Neurodegeneration", "Deafness", "Dwarfism", "Epilepsy", "Diamond-Blackfan anemia", "Neuropathy", "Chromosomal rearrangement"}
G = G0[~G0.keyword.isin(DIS)].head(14); EXR = [("f", G0[G0.keyword == "Apoptosis"].iloc[0]), ("g", G0[G0.keyword == "DNA replication"].iloc[0])]
gx0, gx1 = 250, 550; y0g, y1g = T + 14, T + 14 + GH - 22; gap = 8; rh = (y1g - y0g - gap) / 16; glo, ghi = 0, 0.8
for t in (0, 0.2, 0.4, 0.6, 0.8): xx = gx0 + (t - glo) / (ghi - glo) * (gx1 - gx0); line(xx, y0g, xx, y1g); text(xx, y1g + 16, "%.1f" % t, FS_TICK, anchor="middle")
line(gx0, y0g, gx0, y1g, "#777777", 1.3); line(gx0, y1g, gx1, y1g, "#777777", 1.3); text((gx0 + gx1) / 2, y1g + 34, "Human true pairs recovered, by UniProt keyword category", FS_AX, anchor="middle", fill=INK)
def drow(cy, row, bold=False, tag=None):
    if tag: add(f'<rect x="{20}" y="{cy - rh / 2 - 1:.1f}" width="{gx1 - 20 + 12}" height="{rh + 2:.1f}" rx="4" fill="#ECE9F8"/>'); circle(gx1 + 24, cy, 8, COL["PanPPI"]); text(gx1 + 24, cy + 4.3, tag, 11.5, "bold", anchor="middle", fill="#ffffff")
    text(gx0 - 8, cy + 4.2, row.keyword, 12, "bold" if bold else "normal", anchor="end", fill=INK)
    for v, lo_, hi_, c, rr in ((row.afm, row.afm_lo, row.afm_hi, COL["AFM flag"], 3.8), (row.lg_call, None, None, COL["LoGoPPI"], 3.8), (row.pan_call, row.pan_call_lo, row.pan_call_hi, COL["PanPPI"], 4.8)):
        px = gx0 + v / ghi * (gx1 - gx0)
        if lo_ is not None: line(gx0 + lo_ / ghi * (gx1 - gx0), cy, gx0 + hi_ / ghi * (gx1 - gx0), cy, c, 2.0)
        circle(px, cy, rr, c, "#ffffff", 1.1)
    if pd.notna(getattr(row, "pan_clean", np.nan)) and getattr(row, "n_clean", 0) >= 150:
        pa = gx0 + row.pan_call / ghi * (gx1 - gx0); pc = gx0 + row.pan_clean / ghi * (gx1 - gx0); line(pc, cy, pa, cy, COL["PanPPI"], 1.2, "3 2"); circle(pc, cy, 4.8, "none", COL["PanPPI"], 1.7)
for r_, row in enumerate(G.itertuples()): drow(y0g + (r_ + 0.5) * rh, row)
for q, (tg, row) in enumerate(EXR): drow(y0g + 14 * rh + gap + (q + 0.5) * rh, row, True, tg)
# ---------------------------------------------------------------- h, i: worked examples
BB = pd.read_csv(H + "out3/human/cand_scores.tsv.gz", sep="\t"); BB = BB[(BB.variant == "A") & (BB.level == "L0") & (BB.tier == "pdb")]; BB["pan"] = np.mean([0.5 * (BB[f"g_{s}"] + BB[f"m_{s}"]) for s in ("s42", "s7", "s141")], axis=0)
NEG = BB[BB.label == 0]; PR = BB[BB.label == 1]; idsl = [l.strip() for l in open(F2 + "annot/human_ids.txt") if l.strip()]; nneg = format(int(round(len(NEG) / NEG.rep.nunique(), -2)), ",")
def pct(vals, v): vals = np.asarray(vals); return 100 * ((vals < v).mean() + 0.5 * (vals == v).mean())
def inset(bx, tag, title, pdbfile, chains, accs, cap, top):
    letter(633 if bx > 300 else 6, top + 20, tag); text(bx + 10, top + 16, title, 13, "bold", fill=INK)
    elems, labs, near, T_, P_, A_, B_ = sp.render(pdbfile, bx + 95, top + 84, 106, chA=chains[0], chB=chains[1], ribbon_partner=True)
    for e in elems: add(e)
    text(bx + 100, top + 152, cap, 11, anchor="middle")
    ia, ib = sorted((idsl.index(accs[0]), idsl.index(accs[1]))); row = PR[(PR.i == ia) & (PR.j == ib)].iloc[0]
    strips = [("PanPPI", COL["PanPPI"], pct(NEG.pan, row.pan), True), ("LoGoPPI", COL["LoGoPPI"], pct(NEG.lg, row.lg), False), ("AFM ipTM %.2f (%s)" % (row.iptm, "called" if row.afm_pass else "not called"), COL["AFM flag"], pct(NEG.iptm, row.iptm), False)]
    ax0, ax1 = bx + 310, bx + 510; ys = [top + 60 + 28 * k for k in range(3)]
    add(f'<rect x="{ax0 + 0.95 * (ax1 - ax0)}" y="{top + 31}" width="{0.05 * (ax1 - ax0)}" height="{ys[-1] - top - 31 + 14}" fill="#E3DFF6"/>'); text(ax0 + 0.975 * (ax1 - ax0), top + 26, "called", 11, "bold", anchor="middle", fill=COL["PanPPI"])
    for k, (nm, c, p, b_) in enumerate(strips):
        y = ys[k]; text(ax0 - 10, y + 4.5, nm, 12.5, "bold" if b_ else "normal", anchor="end", fill=INK); line(ax0, y, ax1, y, "#bbbbbb", 1.5)
        for t in (0, 50, 95, 100): xx = ax0 + t / 100 * (ax1 - ax0); line(xx, y - 4, xx, y + 4, "#999999", 1.1)
        px = ax0 + p / 100 * (ax1 - ax0); circle(px, y, 7 if b_ else 6, c, "#ffffff", 1.6); text(px, y - 11, "%.0f%%" % p, 12, "bold" if b_ else "normal", anchor="middle", fill=INK)
    text((ax0 + ax1) / 2, ys[-1] + 25, "percentile among %s negatives" % nneg, 11.5, anchor="middle")
S3 = F2 + "structs/"
inset(670, "f", "BID – MCL1", S3 + "2kbw_m1.pdb", ("A", "B"), ("Q07820", "P55957"), "PDB 2KBW", TOPS[2])
inset(670, "g", "MCM6 – CDT1", S3 + "2le8_m1.pdb", ("A", "B"), ("Q14566", "Q9H211"), "PDB 2LE8", TOPS[2] + 156)
body = "\n".join(out)
open(F2 + "fig2_v17.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}"><rect width="{W}" height="{HT}" fill="#ffffff"/>\n{body}\n</svg>'); print("wrote fig2_v17.svg", W, "x", HT)
