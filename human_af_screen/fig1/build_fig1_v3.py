#!/usr/bin/env python3
"""Figure 1, version 3 (Nature Methods style). Panels in reading order: a schematic, b standard cross-species, c Bernett, d Bernett-style cross-species, e runtime, f branch slope charts.
Palette: validated categorical set (indigo for PanPPI); the schematic (panel a) is recoloured to the same palette (RECOLOR_A=False keeps the original colours). Run from this folder: python3 build_fig1_v3.py -> fig1_v3.svg
Bars start at zero; PanPPI/TUnA seeds are open circles on the bar tops. Data: CSV tables in this folder, ../xsp_bern/res/lg_branch/*.csv, Bernett test labels from the scratchpad scores."""
import re, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
H = "<ROOT>/human_af_screen/"; B = "/private/tmp/claude-1618750309/-Users-<user>-oliva-hpc/a067ef54-32be-4e05-a495-9d04f41d958f/scratchpad/"
RECOLOR_A = True; W, HT = 1220, 1445; INK, INK2, GRID = "#111111", "#444444", "#e8e8e8"
COL = {"PanPPI": "#4A3FA0", "LoGoPPI": "#E0566A", "PLM-interact": "#D9A21B", "PPLM-PPI": "#3B9AC8", "ProteomeLM-PPI": "#2E9E5B", "TUnA": "#B04C9E"}
TOOLS = ["PanPPI", "LoGoPPI", "PLM-interact", "PPLM-PPI", "ProteomeLM-PPI", "TUnA"]; KEY = {"PanPPI": "PanPPI blend"}
GLOBAL_C, MAXSIM_C = "#1F5A99", "#0F8B8D"
FS_TICK, FS_AX, FS_LAB, FS_LEG, FS_VAL = 13, 14, 14.5, 14.5, 12.5
out = []; add = out.append
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def text(x, y, s, size=FS_TICK, weight="normal", anchor="start", fill=INK2, italic=False, rot=None):
    st = ' font-style="italic"' if italic else ""; tr = f' transform="rotate({rot} {x:.1f} {y:.1f})"' if rot is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}"{st}{tr}>{esc(s)}</text>')
def rich(x, y, parts, size, anchor="middle", fill=INK):
    spans = "".join(f'<tspan font-style="{"italic" if it else "normal"}">{esc(s)}</tspan>' for s, it in parts)
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{fill}">{spans}</text>')
def line(x1, y1, x2, y2, stroke=GRID, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""; add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>')
def circle(cx, cy, r, fill, stroke="none", sw=1.0): add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def letter(x, y, s): text(x, y, s, 26, "bold", fill=INK)
def axes(x0, x1, y0, y1, lo, hi, ticks, label, fmt, ytitle=True, grid=True):
    for t in ticks:
        y = y1 - (t - lo) / (hi - lo) * (y1 - y0)
        if grid: line(x0, y, x1, y)
        text(x0 - 7, y + 4.5, fmt % t, FS_TICK, anchor="end")
    line(x0, y0, x0, y1, "#777777", 1.3); line(x0, y1, x1, y1, "#777777", 1.3)
    if ytitle: text(x0 - 50, (y0 + y1) / 2, label, FS_AX, anchor="middle", rot=-90, fill=INK)
def yv(v, lo, hi, y0, y1): return y1 - (v - lo) / (hi - lo) * (y1 - y0)
def bar(cx, v, tool, bw, lo, hi, y0, y1, seeds=None, size=FS_VAL, bold=False):
    top = yv(v, lo, hi, y0, y1); base = yv(max(lo, 0), lo, hi, y0, y1); add(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw}" height="{base - top:.1f}" fill="{COL[tool]}"/>')
    hp = top
    for s in (seeds or []): ys = yv(s, lo, hi, y0, y1); circle(cx, ys, 2.9, "#ffffff", COL[tool], 1.6); hp = min(hp, ys)
    text(cx, hp - 7, (".%03d" % round(v * 1000)) if v < 1 else "1.00", size, "bold" if bold else "normal", anchor="middle", fill=INK)
SETS = ["Yeast", "Mouse", "Fly", "Worm", "E. coli"]; dmap = {"Yeast": "yeast", "Mouse": "mouse", "Fly": "fly", "Worm": "worm", "E. coli": "ecoli"}
def setlabel(cx, y, nm):
    if nm == "E. coli": rich(cx, y, [("E. coli", True)], FS_LAB)
    else: text(cx, y, nm, FS_LAB, anchor="middle", fill=INK)
# ------------------------------------------------------------------ data
S = pd.read_csv(H + "fig1/panel_b_standard_per_seed.csv"); Sm = pd.read_csv(H + "fig1/panel_b_standard_summary.csv"); Dd = pd.read_csv(H + "fig1/panel_d_bernett_style_crossspecies.csv")
def seedvals(key, st): return S[(S.tool == key) & (S.set == st)].aupr.tolist()
# ------------------------------------------------------------------ panel a (schematic, recoloured to the figure palette)
svg_a = open("<ROOT>/ref/fig1a_architecture_v4 copy.svg").read()
svg_a = re.sub(r"<\?xml[^>]*\?>\s*", "", svg_a); svg_a = re.sub(r"<!--.*?-->\s*", "", svg_a, flags=re.S)
svg_a = re.sub(r'\s*<text class="st39"[^>]*>\s*<tspan[^>]*>Lorem ipsum</tspan>\s*</text>', "", svg_a); svg_a = svg_a.replace('<svg id="Layer_1"', '<svg id="panel_a" x="0" y="0" width="1220" height="470"', 1).replace('version="1.1" viewBox', 'viewBox', 1)
if RECOLOR_A:
    for a, b in {"#2b6cb0": GLOBAL_C, "#e6f0fa": "#E3ECF6", "#123a63": "#153B66", "#1f8a78": MAXSIM_C, "#e3f4f1": "#DDF1F1", "#0d4a3f": "#0B4B4C", "#5e4b8b": COL["PanPPI"], "#ede7f6": "#E9E6F5", "#2e2150": "#231B5C"}.items(): svg_a = svg_a.replace(a, b)
# ------------------------------------------------------------------ shared legend
ly = 503; lx = 24
for tool in TOOLS:
    add(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" rx="2" fill="{COL[tool]}"/>'); text(lx + 20, ly, tool, FS_LEG, "bold" if tool == "PanPPI" else "normal", fill=INK); lx += 20 + 8.4 * len(tool) + 24
circle(lx + 6, ly - 5, 3.6, "#ffffff", INK2, 1.7); text(lx + 17, ly, "individual seed", FS_LEG, fill=INK)
# ------------------------------------------------------------------ b: standard cross-species, full width
top = 525; y0, y1 = top + 58, top + 248; letter(20, top + 28, "b"); x0, x1 = 82, 1195
axes(x0, x1, y0, y1, 0, 1.0, [0, 0.2, 0.4, 0.6, 0.8, 1.0], "AUPR", "%.1f"); gw = (x1 - x0) / 5; sp = 34; bw = 30
for i, st in enumerate(SETS):
    cx = x0 + (i + 0.5) * gw; setlabel(cx, y1 + 24, st)
    for j, tool in enumerate(TOOLS):
        vals = seedvals(KEY.get(tool, tool), st)
        if vals: bar(cx + (j - 2.5) * sp, float(np.mean(vals)), tool, bw, 0, 1.0, y0, y1, seeds=vals if len(vals) > 1 else None, bold=(tool == "PanPPI"))
# ------------------------------------------------------------------ c and d (row 3)
top = 845; y0, y1 = top + 58, top + 248
letter(20, top + 28, "c"); cx0, cx1 = 82, 392; axes(cx0, cx1, y0, y1, 0, 0.8, [0, 0.2, 0.4, 0.6, 0.8], "AUPR", "%.1f"); gw2 = (cx1 - cx0) / 4
for i, (tool, vals) in enumerate([("PanPPI", [0.7117, 0.7086, 0.7041]), ("LoGoPPI", [0.6843]), ("PLM-interact", [0.6880]), ("TUnA", [0.6911])]):
    cx = cx0 + (i + 0.5) * gw2; bar(cx, float(np.mean(vals)), tool, 52, 0, 0.8, y0, y1, seeds=vals if len(vals) > 1 else None, size=13.5, bold=(tool == "PanPPI"))
    text(cx + 5, y1 - 9, tool, 13.5, "bold" if i == 0 else "normal", fill="#ffffff" if tool in ("PanPPI", "LoGoPPI", "TUnA") else INK, rot=-90)
letter(440, top + 28, "d"); x0, x1 = 520, 1195; axes(x0, x1, y0, y1, 0, 1.0, [0, 0.2, 0.4, 0.6, 0.8, 1.0], "AUPR", "%.1f"); gw = (x1 - x0) / 5; sp = 30; bw = 26
dt = [("PanPPI", None), ("LoGoPPI", "LoGoPPI-B blend"), ("PLM-interact", "PLM-interact-B"), ("TUnA", "TUnA-B")]
for i, st in enumerate(SETS):
    cx = x0 + (i + 0.5) * gw; setlabel(cx, y1 + 24, st); sub = Dd[Dd.set == dmap[st]]
    for j, (tool, key) in enumerate(dt):
        vals = [float(v) for v in sub[sub.tool.isin(["PanPPI blend s7", "PanPPI blend s141", "PanPPI blend s42"])].aupr] if tool == "PanPPI" else [float(sub[sub.tool == key].aupr.iloc[0])]
        bar(cx + (j - 1.5) * sp, float(np.mean(vals)), tool, bw, 0, 1.0, y0, y1, seeds=vals if len(vals) > 1 else None, size=12, bold=(tool == "PanPPI"))
# ------------------------------------------------------------------ e: runtime (row 4, left)
top = 1165; y0, y1 = top + 58, top + 228; letter(20, top + 28, "e"); ex0, ex1 = 175, 540; lo, hi = 2.0, 5.0
def xl(sec): return ex0 + (np.log10(sec) - lo) / (hi - lo) * (ex1 - ex0)
for t in (100, 1000, 10000, 100000):
    x = xl(t); line(x, y0, x, y1, GRID)
    add(f'<text x="{x:.1f}" y="{y1 + 23}" font-family="Arial, Helvetica, sans-serif" font-size="13" text-anchor="middle" fill="{INK2}">10<tspan baseline-shift="super" font-size="9">{int(np.log10(t))}</tspan></text>')
line(ex0, y1, ex1, y1, "#777777", 1.3); line(ex0, y0, ex0, y1, "#777777", 1.3); text((ex0 + ex1) / 2, y1 + 45, "seconds to score 14 million pairs (one H200)", FS_AX - 1, anchor="middle", fill=INK)
rows = [("ProteomeLM-PPI", [250.0], "4.2 min"), ("LoGoPPI", [942.3], "15.7 min"), ("PanPPI", [1073.5, 1119.7, 1109.9], "17.9–18.7 min"), ("TUnA", [51120.0], "14.1–14.3 h")]
for k, (tool, vals, lab) in enumerate(rows):
    y = y0 + 22 + k * 42; text(ex0 - 9, y + 5, tool, FS_LAB - 1, "bold" if tool == "PanPPI" else "normal", anchor="end", fill=INK); xe = xl(float(np.mean(vals)))
    add(f'<rect x="{ex0}" y="{y - 10}" width="{xe - ex0:.1f}" height="20" fill="{COL[tool]}"/>')
    if tool == "TUnA": text(xe - 8, y + 5, lab, 13, "bold", anchor="end", fill="#ffffff")
    else: text(xe + 8, y + 5, lab, 13, "bold" if tool == "PanPPI" else "normal", fill=INK)
# ------------------------------------------------------------------ f: slope charts (row 4, right)
letter(610, top + 28, "f"); y0f, y1f = top + 66, top + 228; lo, hi = 0.5, 1.0
bl = {}
for st in SETS:
    g = Sm[(Sm.set == st) & (Sm.tool == "PanPPI global")].aupr.iloc[0]; m = Sm[(Sm.set == st) & (Sm.tool == "PanPPI MaxSim")].aupr.iloc[0]; bl[("PanPPI", st)] = (g, m)
    lg = pd.read_csv(H + f"xsp_bern/res/lg_branch/{dmap[st]}.csv", dtype={"query": str, "text": str}); u = pd.read_csv(B + f"unc/pairs_{dmap[st]}.tsv.gz", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label"]).rename(columns={"a": "query", "b": "text"})
    mm = lg.merge(u, on=["query", "text"]); assert len(mm) == len(lg); bl[("LoGoPPI", st)] = (AP(mm.label, mm.global_score), AP(mm.label, mm.maxsim_score))
pc = pd.read_csv(H + "fig1/panel_c_bernett_panppi_seeds.csv"); bl[("PanPPI", "Bernett")] = (float(pc[pc.tool == "PanPPI global"].aupr_mean.iloc[0]), float(pc[pc.tool == "PanPPI MaxSim"].aupr_mean.iloc[0]))
lgb = pd.read_csv(H + "xsp_bern/res/lg_branch/bernett.csv", dtype={"query": str, "text": str}); sb = pd.read_csv(B + "bern/s42.tsv", sep="\t", dtype={"a": str, "b": str}, usecols=["a", "b", "label"])
kk = lambda a, b: np.where(a < b, a + "|" + b, b + "|" + a); lgb["k"] = kk(lgb["query"].values, lgb["text"].values); sb["k"] = kk(sb.a.values, sb.b.values); sb = sb.drop_duplicates("k"); mb = lgb.drop_duplicates("k").merge(sb[["k", "label"]], on="k")
bl[("LoGoPPI", "Bernett")] = (AP(mb.label, mb.global_score), AP(mb.label, mb.maxsim_score))
for k in range(6): line(660, yv(0.5 + 0.1 * k, lo, hi, y0f, y1f), 1195, yv(0.5 + 0.1 * k, lo, hi, y0f, y1f), GRID)
for k in range(6): text(653, yv(0.5 + 0.1 * k, lo, hi, y0f, y1f) + 4.5, "%.1f" % (0.5 + 0.1 * k), FS_TICK, anchor="end")
text(612, (y0f + y1f) / 2, "AUPR", FS_AX, anchor="middle", rot=-90, fill=INK)
for tool, xg, xm in (("PanPPI", 700, 830), ("LoGoPPI", 960, 1070)):
    text((xg + xm) / 2, y0f - 24, tool, FS_LAB, "bold", anchor="middle", fill=COL[tool])
    text(xg, y1f + 22, "Global", 13.5, "bold", anchor="middle", fill=GLOBAL_C); text(xm, y1f + 22, "MaxSim", 13.5, "bold", anchor="middle", fill=MAXSIM_C)
    labs = []
    for st in SETS + ["Bernett"]:
        g, m = bl[(tool, st)]; yg, ym = yv(g, lo, hi, y0f, y1f), yv(m, lo, hi, y0f, y1f)
        add(f'<line x1="{xg}" y1="{yg:.1f}" x2="{xm}" y2="{ym:.1f}" stroke="{COL[tool]}" stroke-width="2.3" {"stroke-dasharray=\"6 4\"" if st == "Bernett" else ""}/>')
        circle(xg, yg, 4.8, GLOBAL_C, "#ffffff", 1.2); circle(xm, ym, 4.8, MAXSIM_C, "#ffffff", 1.2); labs.append([ym, st])
    labs.sort(); ys = [l[0] for l in labs]
    for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + 13.5)
    for (ym, st), yl in zip(labs, ys): rich(xm + 10, yl + 4.5, [(st, st == "E. coli")], 12.5, "start", INK2)
body = "\n".join(out)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {HT}" width="{W}" height="{HT}">\n<rect width="{W}" height="{HT}" fill="#ffffff"/>\n{svg_a}\n{body}\n</svg>\n'
open(H + "fig1/fig1_v3.svg", "w").write(svg); print("wrote fig1_v3.svg", len(svg), "bytes")
print("branch AUPR (global, MaxSim):", {k: tuple(round(float(x), 3) for x in v) for k, v in bl.items()})
