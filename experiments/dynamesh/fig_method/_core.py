import sys, math, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from voxelize import load_obj, sample_surface, voxelize, rot
from iso import iso_axes, cube_faces, pts
from palette import *
from tokens import union_bbox, frame_cells, tint_cell, ramp_weights
from svgkit import *

W, H = 1180, 790
FR = (89, 90, 91)                     # the real MCFM window: offsets (-1, 0, +1)
TINTS = (C1, C2, C3)
NG = 12                               # illustrative token grid; real cond = 1029 tokens

# ───────────────────────────── token data from the real frames ────────────────
BBOX = union_bbox(FR)
CELLS = [frame_cells(i, BBOX, NG) for i in FR]

def grid_svg(x, y, size, cellcol, gap=0.0, rx=0.8, line='#FFFFFF', lw=0.55):
    c = size / NG
    o = [f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" rx="2" fill="#ffffff"/>']
    for r in range(NG):
        for k in range(NG):
            o.append(f'<rect x="{x + k*c:.2f}" y="{y + r*c:.2f}" width="{c-gap:.2f}" '
                     f'height="{c-gap:.2f}" rx="{rx}" fill="{hexc(cellcol(r,k))}" '
                     f'stroke="{line}" stroke-width="{lw}"/>')
    o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" rx="2" '
             f'fill="none" stroke="#9FB0CC" stroke-width="1.1"/>')
    return '\n'.join(o)

def frame_grid(n):
    lum, obj = CELLS[n]
    return lambda r, k: tint_cell(lum[r, k], obj[r, k], TINTS[n])

def blend_grid():
    def f(r, k):
        w = ramp_weights(r, k, NG, NG)
        acc = (0.0, 0.0, 0.0)
        for n in range(3):
            lum, obj = CELLS[n]
            c = tint_cell(lum[r, k], obj[r, k], TINTS[n])
            acc = tuple(acc[i] + w[n] * c[i] for i in range(3))
        return acc
    return f

# ───────────────────────────── voxel Spot ─────────────────────────────────────
SPOT = '/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_star/mesh/spot.obj'
_V, _F = load_obj(SPOT)
VOX, _ = voxelize(sample_surface(_V, _F, 600_000) @ rot(135, 'y').T, 13)
_d = VOX[:, 0] + VOX[:, 1] + VOX[:, 2]
DEPTH = (_d - _d.min()) / (_d.max() - _d.min())          # 1 = nearest camera
_u = VOX[:, 0] - VOX[:, 2]
UAX = (_u - _u.min()) / (_u.max() - _u.min())            # screen-x, for the hue ramp
NEUTRAL = (0.745, 0.775, 0.825)

def smoothstep(a, b, t):
    t = min(1.0, max(0.0, (t - a) / (b - a)))
    return t * t * (3 - 2 * t)

# body axis: UAX = 0 at the HEAD (screen-left at yaw 135), 1 at the REAR
_j = VOX[:, 1].astype(float)
JN = (_j - _j.min()) / (_j.max() - _j.min())        # 0 = hooves, 1 = top of head/back
NEUTRAL = (0.745, 0.775, 0.825)
GLASS   = (0.62, 0.68, 0.80)

def hue_at(u):
    """3-colour blend ramp, driven by height so it is independent of the
    head->rear response ramp."""
    w1 = max(0.0, 1 - 2 * u); w3 = max(0.0, 2 * u - 1); w2 = 1 - w1 - w3
    return tuple(w1 * C1[i] + w2 * C2[i] + w3 * C3[i] for i in range(3))

def voxel_svg(x, y, boxw, mode, s=10.0):
    ex, ey, ez = iso_axes(s)
    order = np.argsort(_d, kind='stable')
    poly, xs, ys = [], [], []
    for idx in order:
        i, j, k = VOX[idx]
        front = 1.0 - UAX[idx]                       # 1 at the head, 0 at the rear
        col_hue = hue_at(1.0 - JN[idx])              # blue on top -> amber at the legs
        if mode == 'ca':                             # after cross-attention
            sat = smoothstep(0.25, 0.85, front)
            a   = 0.26 + 0.50 * sat
            col = mix(NEUTRAL, col_hue, sat)
            edge, ew, eo = '#FFFFFF', 0.45, min(1.0, a + 0.20)
        elif mode == 'sa':                           # after self-attention
            sat, a = 1.0, 0.76
            col = col_hue
            edge, ew, eo = '#FFFFFF', 0.45, 0.95
        else:                                        # bare voxel tokens: glass
            sat, a = 0.0, 0.17
            col = GLASS
            edge, ew, eo = '#647AA0', 0.55, 0.62
        for face, sh in zip(cube_faces(i, j, k, ex, ey, ez), (1.0, 0.82, 0.63)):
            c = tuple(min(1.0, v * sh) for v in col)
            poly.append((pts(face), hexc(c), a, edge, ew, eo))
            for px, py in face:
                xs.append(px); ys.append(py)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    sc = boxw / (x1 - x0)
    body = '\n'.join(f'<polygon points="{p}" fill="{c}" fill-opacity="{a:.3f}" '
                      f'stroke="{e}" stroke-width="{w}" stroke-opacity="{o:.2f}"/>'
                      for p, c, a, e, w, o in poly)
    g = (f'<g transform="translate({x:.1f},{y:.1f}) scale({sc:.4f}) '
         f'translate({-x0:.2f},{-y0:.2f})">{body}</g>')
    return g, (y1 - y0) * sc, (x1 - x0) * sc
