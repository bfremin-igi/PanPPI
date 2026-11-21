"""Supplementary Fig. 9 in the drawing style of Figure 2 (same helpers, tool colours, dot-and-interval rows, precision-versus-coverage panel).
a retention forest (like Fig 2b) | b precision of calls vs true pairs found (like Fig 2c) | c retention in top 1% after hub removal | d AUPR on degree-matched sets (like Supp Fig 2). Run from this folder."""
import numpy as np, pandas as pd
W, HT = 1220, 600; INK, INK2, GRID = "#111111", "#444444", "#e8e8e8"
COL = {"PanPPI": "#4A3FA0", "LoGoPPI": "#E0566A", "AFM ipSAE": "#3D4651", "AFM ipTM": "#9AA3AD", "AFM flag": "#B59B7A"}
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
def bar(cx, v, color, bw, lo, hi, y0, y1, op=1.0):
    top = yv(v, lo, hi, y0, y1); base = yv(lo, lo, hi, y0, y1); add(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(base - top, 0):.1f}" fill="{color}" fill-opacity="{op}"/>')
def ci(x0, x1, y, lo, hi, c, w=2.2):
    xa, xb = x0 + lo * (x1 - x0), x0 + hi * (x1 - x0); line(xa, y, xb, y, c, w)
    for xx in (xa, xb): line(xx, y - 2.5, xx, y + 2.5, c, 1.6)
SPS = ["soybean", "arabidopsis", "rat", "rice", "zebrafish"]; NM = {"soybean": "Soybean", "arabidopsis": "Arabidopsis", "rat": "Rat", "rice": "Rice", "zebrafish": "Zebrafish"}
R = pd.read_csv("results/newsp_retention.csv"); C = pd.read_csv("results/analysis_out/newsp/newsp_cells_results.csv"); P = pd.read_csv("results/analysis_out/newsp/newsp_precision.csv").query("group=='all five species'").set_index("item")
# shared legend
ly = 22; x = 330
for nm, lab in (("PanPPI", "PanPPI"), ("LoGoPPI", "LoGoPPI"), ("AFM ipSAE", "AFM ipSAE"), ("AFM ipTM", "AFM ipTM"), ("AFM flag", "AFM high-confidence call")):
    add(f'<rect x="{x}" y="{ly - 11}" width="13" height="13" rx="2" fill="{COL[nm]}"/>'); text(x + 18, ly, lab, FS_LEG, "bold" if nm == "PanPPI" else "normal", fill=INK); x += 32 + 7.6 * len(lab)
# a: retention forest, four fractions
T = 36; letter(6, T + 20, "a"); RH = 30; ya = T + 44
for fi, (ttl, fx0, fx1, frac) in enumerate((("Top 0.01%", 128, 232, 1e-4), ("Top 0.1%", 258, 362, 1e-3), ("Top 1%", 388, 492, 1e-2), ("Top 10%", 518, 622, 0.1))):
    text((fx0 + fx1) / 2, T + 24, ttl, 12.5, "bold", anchor="middle", fill=INK); yb = ya + 10 + RH * 4 + 14
    for k, sp in enumerate(SPS):
        y = ya + 10 + RH * k
        if fi == 0: text(fx0 - 12, y + 4, NM[sp], 12, anchor="end", fill=INK)
        line(fx0, y, fx1, y, "#e3e3e3", 1.2); r = R[(R.species == sp) & (R.hub_fraction_removed == 0) & np.isclose(R.fraction, frac)].iloc[0]; yp, yl = y - 5, y + 5
        ci(fx0, fx1, yp, r.PanPPI_lo, r.PanPPI_hi, COL["PanPPI"]); ci(fx0, fx1, yl, r.LoGoPPI_lo, r.LoGoPPI_hi, COL["LoGoPPI"])
        xx = fx0 + r.PanPPI * (fx1 - fx0); circle(xx, yp, 4.4, COL["PanPPI"], "#ffffff", 1.2); circle(fx0 + r.LoGoPPI * (fx1 - fx0), yl, 3.8, COL["LoGoPPI"], "#ffffff", 1.1)
        text(xx + (9 if r.PanPPI < 0.08 else 0), yp - 8, "%.0f%%" % (100 * r.PanPPI), 10.5, "bold", anchor="start" if r.PanPPI < 0.08 else "middle", fill=INK)
    xr = fx0 + frac * (fx1 - fx0); line(xr, ya + 2, xr, yb, "#999999", 1.2, "4 3")
    for t in (0, 0.5, 1.0): xx = fx0 + t * (fx1 - fx0); line(xx, yb, xx, yb + 4, "#777777", 1.2); text(xx, yb + 17, "%.1f" % t, FS_TICK, anchor="middle")
text((128 + 622) / 2, yb + 38, "Fraction of AFM-confident pairs kept in the top-ranked pairs", FS_AX, anchor="middle", fill=INK)
text((128 + 622) / 2, yb + 53, "dashed line: expected by chance", 10.5, anchor="middle", fill=INK2)
# b: precision vs true pairs found (pooled)
RX0, RX1 = 705, 1190; letter(633, T + 20, "b"); y0f, y1f = T + 30, T + 14 + 145 + 16; plo, phi = 0.65, 1.0; axes(RX0, RX1, y0f, y1f, plo, phi, [0.7, 0.8, 0.9, 1.0], "Precision of calls", "%.1f")
xl0, xl1 = np.log(3000), np.log(60000); X = lambda v: RX0 + (np.log(v) - xl0) / (xl1 - xl0) * (RX1 - RX0)
for v, lab in ((5000, "5k"), (10000, "10k"), (20000, "20k"), (40000, "40k")): line(X(v), y0f, X(v), y1f, GRID); text(X(v), y1f + 17, lab, FS_TICK, anchor="middle")
text((RX0 + RX1) / 2, y1f + 40, "True pairs found (log scale, per replicate)", FS_AX, anchor="middle", fill=INK)
def pt(item, col):
    row = P.loc[item]; x_ = X(row.true / 3); yy = yv(row.precision, plo, phi, y0f, y1f); line(x_, yv(row.lo, plo, phi, y0f, y1f), x_, yv(row.hi, plo, phi, y0f, y1f), col, 2.0); return x_, yy
xa, ya_ = pt("afm", COL["AFM flag"]); circle(xa, ya_, 9, COL["AFM flag"], "#ffffff", 1.4)
for tool, c, rr in (("lg", COL["LoGoPPI"], 5.5), ("pan", COL["PanPPI"], 6.5)):
    xm, ym = pt(tool + "_match", c); xt, yt = pt(tool + "_top5", c); line(xm, ym, xt, yt, c, 2.0, "5 3"); circle(xt, yt, rr + 1, c, "#ffffff", 1.4); circle(xm, ym, rr - 1.5 if tool == "pan" else rr, c, "#ffffff", 1.3)
text(xa + 14, ya_ - 9, "AFM high-confidence call", 12, fill=INK)
text(X(P.loc["pan_top5"].true / 3) + 10, yv(P.loc["pan_top5"].precision, plo, phi, y0f, y1f) - 22, "PanPPI: %.1f× the true pairs at %.2f" % (P.loc["pan_top5"].true / P.loc["afm"].true, P.loc["pan_top5"].precision), 12, "bold", anchor="end", fill=INK)
text(X(P.loc["lg_top5"].true / 3) + 11, yv(P.loc["lg_top5"].precision, plo, phi, y0f, y1f) - 3, "LoGoPPI: %.1f× at %.2f" % (P.loc["lg_top5"].true / P.loc["afm"].true, P.loc["lg_top5"].precision), 12, fill=INK)
text((RX0 + RX1) / 2, y1f + 58, "five species pooled; left end of each line: same number of calls as AFM; right end: top 5% of negatives", 11.5, anchor="middle", fill=INK2)
# c: hub removal rows
T2 = 330; letter(6, T2 + 20, "c"); ya = T2 + 44
for fi, (ttl, h) in enumerate((("No hubs removed", 0.0), ("Top 1% removed", 0.01), ("Top 5% removed", 0.05), ("Top 10% removed", 0.10))):
    fx0, fx1 = 128 + 130 * fi, 232 + 130 * fi; text((fx0 + fx1) / 2, T2 + 24, ttl, 12, "bold", anchor="middle", fill=INK); yb = ya + 10 + RH * 4 + 14
    for k, sp in enumerate(SPS):
        y = ya + 10 + RH * k
        if fi == 0: text(fx0 - 12, y + 4, NM[sp], 12, anchor="end", fill=INK)
        line(fx0, y, fx1, y, "#e3e3e3", 1.2); r = R[(R.species == sp) & np.isclose(R.hub_fraction_removed, h) & np.isclose(R.fraction, 0.01)].iloc[0]; yp, yl = y - 5, y + 5
        ci(fx0, fx1, yp, r.PanPPI_lo, r.PanPPI_hi, COL["PanPPI"]); ci(fx0, fx1, yl, r.LoGoPPI_lo, r.LoGoPPI_hi, COL["LoGoPPI"])
        xx = fx0 + r.PanPPI * (fx1 - fx0); circle(xx, yp, 4.4, COL["PanPPI"], "#ffffff", 1.2); circle(fx0 + r.LoGoPPI * (fx1 - fx0), yl, 3.8, COL["LoGoPPI"], "#ffffff", 1.1); text(xx, yp - 8, "%.0f%%" % (100 * r.PanPPI), 10.5, "bold", anchor="middle", fill=INK)
    for t in (0, 0.5, 1.0): xx = fx0 + t * (fx1 - fx0); line(xx, yb, xx, yb + 4, "#777777", 1.2); text(xx, yb + 17, "%.1f" % t, FS_TICK, anchor="middle")
text((128 + 622 + 390) / 2 - 0, yb + 38, "Fraction of AFM-confident pairs kept in the top 1% of pairs", FS_AX, anchor="middle", fill=INK)
# d: AUPR degree-matched
RX0d, RX1d = 705, 1190; letter(633, T2 + 20, "d"); y0, y1 = T2 + 30, T2 + 14 + 145 + 16; axes(RX0d, RX1d, y0, y1, 0.5, 0.8, [0.5, 0.6, 0.7, 0.8], "AUPR (degree-matched)", "%.1f")
k_ = C[C.kept]; cells = [(sp, lv) for sp in SPS for lv in ("L0", "L2") if not k_[(k_.species == sp) & (k_.level == lv)].empty]; gw = (RX1d - RX0d) / len(cells)
tl = [("PanPPI blend aupr", "PanPPI"), ("LoGoPPI aupr", "LoGoPPI"), ("AFM ipSAE aupr", "AFM ipSAE"), ("AFM ipTM aupr", "AFM ipTM")]
for g, (sp, lv) in enumerate(cells):
    cx = RX0d + (g + 0.5) * gw; r_ = k_[(k_.species == sp) & (k_.level == lv)].iloc[0]
    for j, (cn, nm) in enumerate(tl):
        bx_ = cx + (j - 1.5) * (gw / 4.6); bar(bx_, float(r_[cn]), COL[nm], gw / 5.0, 0.5, 0.8, y0, y1); text(bx_, yv(float(r_[cn]), 0.5, 0.8, y0, y1) - 4, ("%.2f" % float(r_[cn])).lstrip("0"), 8.5, "bold" if nm == "PanPPI" else "normal", anchor="middle", fill=INK)
    text(cx, y1 + 16, NM[sp][:3] + ("." if len(NM[sp]) > 3 else ""), 11, anchor="middle", fill=INK); text(cx, y1 + 30, "all" if lv == "L0" else "no homolog", 9.5, anchor="middle")
body = "\n".join(out); open("supp9_v2.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}"><rect width="{W}" height="{HT}" fill="#ffffff"/>\n{body}\n</svg>'); print("ok")
