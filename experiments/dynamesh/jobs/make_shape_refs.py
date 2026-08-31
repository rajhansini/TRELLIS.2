"""make_shape_refs.py -- grey shape references for Q3, at all four cameras.

WHY THIS IS NOT A TRIVIAL SCRIPT
    render_grey.py and calib_cpu_raster.py both hardcode EXT to the training camera.
    Neither takes a yaw or an elevation, so neither can produce the diagA/diagB/diagC
    reference Q3 needs. Adding the rotation means picking a convention, and picking it
    wrong is silent: the render still looks like the object, just from the wrong side,
    and Q3 would then score every method against a bad reference. That is the same
    class of mistake as asking about shape stability instead of shape correctness.

SO THE CONVENTION IS SOLVED, NOT GUESSED
    We already own a correct render of the given mesh at each of the four cameras:
    the `ours` panel, which is produced from that exact mesh by the pipeline itself.
    Its silhouette is therefore ground truth for this rasteriser. Every candidate
    convention (yaw sign, elevation sign, rotation order) is scored by silhouette IoU
    against that panel, and the winner is the one that matches. calib_cpu_raster.py
    settled flip_y the same way rather than guessing; this extends the trick to the
    full rotation.

    A convention that scores < MIN_IOU is refused rather than shipped.

Stages
    solve   -- find the convention on a few objects, print the IoU table
    render  -- write 42 x 4 grey PNGs using the solved convention, and record the
               per-cell IoU so a bad cell is visible rather than silent
"""
import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
VIEWS = {'train': (0.0, 0.0), 'diagA': (45.0, 25.0),
         'diagB': (135.0, -20.0), 'diagC': (225.0, 30.0)}
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXT = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 2.], [0, 0, 0, 1.]])
NEAR, FAR = 0.5, 3.0
LAB, PANEL = 28, 518          # label strip and panel width of the pipeline render
MIN_IOU = 0.90
RES = 384                     # matches the judged videos


def projection():
    r = np.zeros((4, 4))
    r[0, 0] = 2 * _FX; r[1, 1] = 2 * _FX
    r[2, 2] = FAR / (FAR - NEAR); r[2, 3] = NEAR * FAR / (NEAR - FAR); r[3, 2] = 1.0
    return r


def norm(v):
    """Same normalisation render_grey.py applies: unit box, then the y/z swap that
    puts the mesh into the render frame."""
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    t = v[:, 1].copy(); v[:, 1] = -v[:, 2]; v[:, 2] = t
    return v


def _rot(axis, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if axis == 'x': return np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y': return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rotate(V, yaw, elev, yaw_sign, elev_sign, order, up='y'):
    """Yaw about the render frame's up axis, then tilt about the axis orthogonal to
    it. Which axis is 'up' after the pipeline's y/z swap is not assumed -- it is one
    of the things the solve stage searches over."""
    tilt = 'x' if up == 'y' else 'x'
    Ry = _rot(up, yaw * yaw_sign)
    Re = _rot(tilt, elev * elev_sign)
    M = (Re @ Ry) if order == 'ey' else (Ry @ Re)
    return V @ M.T


def raster(V, F, res):
    """Z-buffered barycentric rasteriser. Returns (mask, shaded) both res x res."""
    P = projection()
    vh = np.concatenate([V, np.ones((len(V), 1))], 1)
    clip = vh @ EXT.T @ P.T
    w = np.maximum(clip[:, 3:4], 1e-8)
    ndc = clip[:, :3] / w
    px = (ndc[:, 0] * 0.5 + 0.5) * res
    py = (ndc[:, 1] * 0.5 + 0.5) * res          # flip_y=False, per calib_cpu_raster
    z = ndc[:, 2]
    mask = np.zeros((res, res), bool)
    depth = np.full((res, res), np.inf)
    shade = np.zeros((res, res))
    # face normals in the rotated frame, lit from the camera
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    lam = np.clip(-n[:, 2], 0, 1) * 0.75 + 0.25     # camera looks along +z here
    for f in range(len(F)):
        i0, i1, i2 = F[f]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        xlo, xhi = int(max(0, min(x0, x1, x2))), int(min(res - 1, max(x0, x1, x2)))
        ylo, yhi = int(max(0, min(y0, y1, y2))), int(min(res - 1, max(y0, y1, y2)))
        if xhi < xlo or yhi < ylo:
            continue
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12:
            continue
        xs = np.arange(xlo, xhi + 1) + 0.5
        ys = np.arange(ylo, yhi + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / d
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / d
        c = 1 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        zz = a * z[i0] + b * z[i1] + c * z[i2]
        sub = depth[ylo:yhi + 1, xlo:xhi + 1]
        win = inside & (zz < sub)
        if not win.any():
            continue
        sub[win] = zz[win]
        mask[ylo:yhi + 1, xlo:xhi + 1][win] = True
        shade[ylo:yhi + 1, xlo:xhi + 1][win] = lam[f]
    return mask, shade


def ours_mask(obj, view, res):
    """Silhouette of the pipeline's own render of this mesh at this camera.
    That render comes from the given mesh, so this mask is the ground truth
    against which the rasteriser's convention is calibrated."""
    d = E / f'out/view_{obj}_27g_{view}/frames'
    fs = sorted(d.glob('*.png')) if d.is_dir() else []
    if not fs:
        return None
    im = Image.open(fs[0]).convert('RGB')
    im = im.crop((PANEL, LAB, 2 * PANEL, LAB + PANEL)).resize((res, res), Image.NEAREST)
    a = np.asarray(im).astype(np.int16)
    # pipeline renders on a white ground; anything not near-white is the object
    return (a.max(2) < 245) | (np.ptp(a, axis=2) > 12)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def load_mesh(obj, meshes):
    """The tsv points at <obj>_render_frame.obj, which is ALREADY in the render
    frame -- the same case render_grey.py calls --raw. Applying norm() on top of that
    transforms it twice: measured train-view IoU 0.71 with norm against 0.99 without.
    So the vertices are used as they are."""
    m = trimesh.load(meshes[obj], process=False, force='mesh')
    return np.asarray(m.vertices, np.float64).copy(), np.asarray(m.faces)


def read_meshes():
    """Column 2 is the object and column 4 the mesh. Column 1 is the BATCH LETTER --
    reading it as the object is the bug that destroyed 12 pumpkin_rot cells."""
    out = {}
    for line in (E / 'jobs/rung37_objects.tsv').read_text().splitlines()[1:]:
        p = line.split('\t')
        if len(p) >= 4:
            out[p[1]] = p[3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['solve', 'render'])
    ap.add_argument('--objects', default='spot_lava,chair_moss,octopus_rust')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--conv', default='')
    ap.add_argument('--out', default=str(E / 'out/JUDGE/shape_refs'))
    a = ap.parse_args()

    meshes = read_meshes()
    objs = sorted(meshes) if a.all else [o for o in a.objects.split(',') if o]

    if a.stage == 'solve':
        cands = list(product((1, -1), (1, -1), ('ey', 'ye'), ('y', 'z')))
        print(f'{"yaw":>4s}{"elev":>6s} {"order":>6s} {"up":>4s}  mean IoU vs pipeline render')
        best, scores = None, {}
        for ys, es, order, up in cands:
            vals = []
            for o in objs:
                if o not in meshes:
                    continue
                V, F = load_mesh(o, meshes)
                for v, (yaw, elev) in VIEWS.items():
                    gt = ours_mask(o, v, RES)
                    if gt is None:
                        continue
                    m, _ = raster(rotate(V, yaw, elev, ys, es, order, up), F, RES)
                    vals.append(iou(m, gt))
            if not vals:
                sys.exit('no reference renders found; cannot calibrate')
            mu = float(np.mean(vals))
            scores[f'{ys},{es},{order},{up}'] = mu
            print(f'{ys:>4d}{es:>6d} {order:>6s} {up:>4s}  {mu:.4f}  min={min(vals):.3f}')
            if best is None or mu > scores[best]:
                best = f'{ys},{es},{order},{up}'
        print(f'\nBEST: {best}  IoU={scores[best]:.4f}')
        if scores[best] < MIN_IOU:
            print(f'REFUSED: below {MIN_IOU}. The convention is still wrong, or the '
                  f'reference crop is. Do NOT render against this.')
            sys.exit(1)
        print(f'run:  python3 jobs/make_shape_refs.py render --all --conv {best}')
        return

    if not a.conv:
        sys.exit('render needs --conv from the solve stage')
    ys, es, order, up = a.conv.split(',')
    ys, es = int(ys), int(es)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for o in objs:
        V, F = load_mesh(o, meshes)
        for v, (yaw, elev) in VIEWS.items():
            m, sh = raster(rotate(V, yaw, elev, ys, es, order, up), F, RES)
            img = np.full((RES, RES, 3), 255, np.uint8)
            g = (sh * 255).clip(0, 255).astype(np.uint8)
            img[m] = np.stack([g, g, g], -1)[m]
            Image.fromarray(img).save(out / f'{o}__{v}.png')
            gt = ours_mask(o, v, RES)
            rec[f'{o}|{v}'] = round(iou(m, gt), 4) if gt is not None else None
        print(f'  {o}', flush=True)
    (out / 'iou.json').write_text(json.dumps(rec, indent=1, sort_keys=True))
    vals = [x for x in rec.values() if x is not None]
    bad = {k: x for k, x in rec.items() if x is not None and x < MIN_IOU}
    print(f'\n{len(rec)} refs written to {out}')
    print(f'IoU vs pipeline render: mean {np.mean(vals):.4f}  min {min(vals):.4f}')
    print(f'BELOW {MIN_IOU}: {len(bad)}' + (f'  {list(bad)[:10]}' if bad else '  (none)'))


if __name__ == '__main__':
    main()
