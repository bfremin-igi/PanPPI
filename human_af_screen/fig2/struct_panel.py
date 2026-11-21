"""Renders PDB 3D9U (cIAP1 BIR3 + SMAC AVPIAQ peptide; crystal structure, 2.30 A) as SVG elements, no external software: orthographic projection of the deposited coordinates.
BIRC2 BIR3: translucent atom surface + C-alpha ribbon; SMAC peptide: ball-and-stick; BIR3 residues within 4.5 A of the peptide: side-chain sticks. View chosen by a search for the orientation in which the peptide is least occluded."""
import numpy as np
def parse(path):
    at = []
    for l in open(path):
        if l.startswith("ATOM"): at.append(dict(ch=l[21], res=l[17:20].strip(), num=int(l[22:26]), name=l[12:16].strip(), el=l[76:78].strip() or l[12:14].strip(), x=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return at
def render(path, cx, cy, size, col_pep="#4A3FA0", col_prot="#7A8594", col_contact="#2B2B2B", seed=3, chA="A", chB="B", ribbon_partner=False):
    at = parse(path); A = [a for a in at if a["ch"] == chA]; Bp = [a for a in at if a["ch"] == chB]
    XA = np.array([a["x"] for a in A]); XB = np.array([a["x"] for a in (Bp if not ribbon_partner else [b for b in Bp if b["name"] == "CA"])]); d = np.linalg.norm(XA[:, None] - XB[None], axis=2); near = sorted({A[k]["num"] for k in np.where(d.min(1) < 4.5)[0]})
    ca = np.array([a["x"] for a in A if a["name"] == "CA"]); cen = XB.mean(0); rng = np.random.default_rng(seed); best = None
    for _ in range(3000):
        q = rng.normal(size=(3, 3)); R, _ = np.linalg.qr(q)
        if np.linalg.det(R) < 0: R[:, 0] *= -1
        PA = (XA - cen) @ R; PB = (XB - cen) @ R   # z toward viewer
        occ = 0
        for p in PB:
            m = (np.linalg.norm(PA[:, :2] - p[:2], axis=1) < 2.2) & (PA[:, 2] > p[2] + 0.5); occ += m.sum()
        cap = np.array([b['x'] for b in Bp if b['name'] == 'CA']); pc = (cap - cen) @ R; pext = max(np.linalg.norm(pc[i, :2] - pc[j, :2]) for i in range(len(pc)) for j in range(len(pc))); score = 0.25 * occ - (0 if ribbon_partner else 30) * pext + (0.5 * np.ptp(np.vstack([PA, PB])[:, :2], axis=0).max() if ribbon_partner else 0)
        if best is None or score < best[0]: best = (score, R)
    R = best[1]; P = lambda X: (X - cen) @ R
    allp = np.vstack([P(XA), P(XB)]); lim = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])); sc = size / (lim * 1.05); mid = (allp[:, :2].max(0) + allp[:, :2].min(0)) / 2
    T = lambda p: (cx + (p[0] - mid[0]) * sc, cy - (p[1] - mid[1]) * sc)
    out = []; surf = sorted([(P(a["x"])[2], a) for a in A], key=lambda t: t[0]); zmin, zmax = allp[:, 2].min(), allp[:, 2].max()
    for z, a in surf:
        x, y = T(P(a["x"])); t = (z - zmin) / (zmax - zmin + 1e-9); sh = int(150 + 70 * t); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.75 * sc:.1f}" fill="rgb({sh - 18},{sh - 8},{sh + 8})" fill-opacity="0.20"/>')
    # CA ribbon (Catmull-Rom), depth-sorted segments
    pts = P(ca); seg = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, len(pts) - 1)]
        for s in range(6):
            u0, u1 = s / 6, (s + 1) / 6; f = lambda u: 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3); seg.append((f(u0), f(u1)))
    for a, b in sorted(seg, key=lambda t: (t[0][2] + t[1][2]) / 2):
        (x1, y1), (x2, y2) = T(a), T(b); t = ((a[2] + b[2]) / 2 - zmin) / (zmax - zmin + 1e-9); g = int(70 + 80 * (1 - t)); out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="rgb({g - 10},{g},{g + 25})" stroke-width="{2.4 + 1.6 * t:.1f}" stroke-linecap="round"/>')
    def bonds(atoms, color, w, r, z_sort=True):
        P_ = [P(a["x"]) for a in atoms]; items = []
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if np.linalg.norm(atoms[i]["x"] - atoms[j]["x"]) < 1.9 and atoms[i]["num"] == atoms[j]["num"] or (np.linalg.norm(atoms[i]["x"] - atoms[j]["x"]) < 1.9 and {atoms[i]["name"], atoms[j]["name"]} == {"C", "N"}): items.append(((P_[i][2] + P_[j][2]) / 2, "b", i, j))
        for i in range(len(atoms)): items.append((P_[i][2], "a", i, i))
        for z, k, i, j in sorted(items, key=lambda t: t[0]):
            if k == "b": (x1, y1), (x2, y2) = T(P_[i]), T(P_[j]); out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')
            elif r: x, y = T(P_[i]); out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>')
    bonds([a for a in A if a["num"] in near and a["name"] not in ("N", "C", "O")] , col_contact, 2.0 * (size / 330) ** 0.5, 0)
    k = (size / 330) ** 0.5
    if ribbon_partner:
        pb = P(np.array([a["x"] for a in Bp if a["name"] == "CA"])); segs = sorted([(pb[i], pb[i + 1]) for i in range(len(pb) - 1)], key=lambda t: t[0][2] + t[1][2])
        for a_, b_ in segs: (x1_, y1_), (x2_, y2_) = T(a_), T(b_); out.append(f'<line x1="{x1_:.1f}" y1="{y1_:.1f}" x2="{x2_:.1f}" y2="{y2_:.1f}" stroke="{col_pep}" stroke-width="{3.6 * k + 1.4:.1f}" stroke-linecap="round"/>')
    else: bonds(Bp, col_pep, 3.6 * k, 3.4 * k)
    # peptide labels: residue letters at CA, offset away from the centroid
    one = {"ALA": "A", "VAL": "V", "PRO": "P", "ILE": "I", "GLN": "Q"}; labs = []
    for a in Bp:
        if a["name"] == "CA": x, y = T(P(a["x"])); labs.append((x, y, one.get(a["res"], a["res"]), a["num"]))
    return out, labs, near, T, P, A, Bp
