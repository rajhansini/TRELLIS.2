import sys, math, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from voxelize import load_obj, rot, voxelize_exact, cull_hidden
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
# Every panel draws the SAME cell set. The figure claims the voxel structure is
# fixed and only appearance changes, so a voxel may never be encoded by fading
# out — a faded cube on white paper reads as a missing cube, which is the exact
# opposite of the claim. Response is carried by hue and saturation ONLY.
SPOT = '/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_star/mesh/spot.obj'
_V, _F = load_obj(SPOT)
_P = _V @ rot(135, 'y').T
VOX_ALL, VOX_DIMS = voxelize_exact(_P, _F, 13)   # 484 cells, no sampling holes
VOX = cull_hidden(VOX_ALL)                       # 317 the camera can actually see

def _norm(a, full):
    """Normalise against the FULL cell set so culling cannot shift a ramp."""
    return (a - full.min()) / (full.max() - full.min())

_d = VOX[:, 0] + VOX[:, 1] + VOX[:, 2]
DEPTH = _norm(_d, VOX_ALL[:, 0] + VOX_ALL[:, 1] + VOX_ALL[:, 2])
UAX = _norm(VOX[:, 0] - VOX[:, 2], VOX_ALL[:, 0] - VOX_ALL[:, 2])   # screen-x: 0 = head
JN = _norm(VOX[:, 1].astype(float), VOX_ALL[:, 1].astype(float))    # 0 = hooves, 1 = back

NEUTRAL = (0.700, 0.730, 0.790)     # "no response yet" — light, but never invisible
SLATE   = (0.615, 0.672, 0.782)     # bare voxel tokens: structure, no appearance
SHADE   = (1.0, 0.82, 0.63)         # top / right / left face

def smoothstep(a, b, t):
    t = min(1.0, max(0.0, (t - a) / (b - a)))
    return t * t * (3 - 2 * t)

def hue_at(u):
    """3-colour blend ramp driven by HEIGHT, so it stays independent of the
    head->rear response ramp and the two cannot be read as one signal."""
    w1 = max(0.0, 1 - 2 * u); w3 = max(0.0, 2 * u - 1); w2 = 1 - w1 - w3
    return tuple(w1 * C1[i] + w2 * C2[i] + w3 * C3[i] for i in range(3))

def voxel_svg(x, y, boxw, mode, s=10.0):
    ex, ey, ez = iso_axes(s)
    order = np.argsort(_d, kind='stable')            # painter's: back to front
    poly, xs, ys = [], [], []
    for idx in order:
        i, j, k = VOX[idx]
        if mode == 'ca':                             # after cross-attention
            sat = smoothstep(0.25, 0.85, 1.0 - UAX[idx])     # 1 at the head
            col = mix(NEUTRAL, hue_at(1.0 - JN[idx]), sat)
            edge, ew = '#FFFFFF', 0.45
        elif mode == 'sa':                           # after self-attention
            col = hue_at(1.0 - JN[idx])
            edge, ew = '#FFFFFF', 0.45
        else:                                        # bare voxel tokens
            col = SLATE
            edge, ew = '#FFFFFF', 0.45
        for face, sh in zip(cube_faces(i, j, k, ex, ey, ez), SHADE):
            c = tuple(min(1.0, v * sh) for v in col)
            poly.append((pts(face), hexc(c), edge, ew))
            for px, py in face:
                xs.append(px); ys.append(py)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    sc = boxw / (x1 - x0)
    body = '\n'.join(f'<polygon points="{p}" fill="{c}" stroke="{e}" '
                     f'stroke-width="{w}" stroke-linejoin="round"/>'
                     for p, c, e, w in poly)
    g = (f'<g transform="translate({x:.1f},{y:.1f}) scale({sc:.4f}) '
         f'translate({-x0:.2f},{-y0:.2f})">{body}</g>')
    return g, (y1 - y0) * sc, (x1 - x0) * sc
