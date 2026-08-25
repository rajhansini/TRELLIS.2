"""calib_cpu_raster.py -- pin the CPU rasteriser to nvdiffrast's actual convention.

check_vase_align_cpu.py guessed the image-y direction and got it wrong, which made
every known-good object score ~0.2-0.5 IoU below its recorded value and produced a
false FAIL on the vase. Rather than guess again, both conventions are rasterised
and scored against out/gt_targets_<obj>/render_mask.npy -- the mask nvdiffrast
itself wrote for a mesh that is already in the render frame (--raw).
"""
import math
from pathlib import Path
import numpy as np, trimesh

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
_FX = 1.0/(2.0*math.tan(math.radians(40.0/2)))
EXT = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 2.], [0, 0, 0, 1.]])
NEAR, FAR = 0.5, 3.0


def projection():
    r = np.zeros((4, 4))
    r[0, 0] = 2*_FX; r[1, 1] = 2*_FX
    r[0, 2] = 0.0;   r[1, 2] = 0.0
    r[2, 2] = FAR/(FAR-NEAR); r[2, 3] = NEAR*FAR/(NEAR-FAR); r[3, 2] = 1.0
    return r


def raster(V, F, res, flip_y):
    P = projection() @ EXT
    vh = np.c_[V, np.ones(len(V))]
    clip = vh @ P.T
    w = clip[:, 3].copy(); w[np.abs(w) < 1e-12] = 1e-12
    ndc = clip[:, :3]/w[:, None]
    px = (ndc[:, 0]*0.5+0.5)*res
    py = ((-ndc[:, 1] if flip_y else ndc[:, 1])*0.5+0.5)*res
    Pp = np.stack([px, py], 1)
    m = np.zeros((res, res), bool)
    for i0, i1, i2 in F:
        p0, p1, p2 = Pp[i0], Pp[i1], Pp[i2]
        x0 = int(max(0, np.floor(min(p0[0], p1[0], p2[0])))); x1 = int(min(res-1, np.ceil(max(p0[0], p1[0], p2[0]))))
        y0 = int(max(0, np.floor(min(p0[1], p1[1], p2[1])))); y1 = int(min(res-1, np.ceil(max(p0[1], p1[1], p2[1]))))
        if x1 < x0 or y1 < y0: continue
        d = (p1[0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[1]-p0[1])
        if abs(d) < 1e-12: continue
        xs, ys = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
        w1 = ((xs-p0[0])*(p2[1]-p0[1])-(ys-p0[1])*(p2[0]-p0[0]))/d
        w2 = ((p1[0]-p0[0])*(ys-p0[1])-(p1[1]-p0[1])*(xs-p0[0]))/d
        ins = (w1 >= -1e-6) & (w2 >= -1e-6) & ((1-w1-w2) >= -1e-6)
        if ins.any(): m[y0:y1+1, x0:x1+1] |= ins
    return m


CASES = [('spot_lava',        'data/spot_lava/mesh/spot_render_frame.obj'),
         ('teapot_porcelain', 'data/teapot_porcelain/mesh/teapot_porcelain_render_frame.obj')]
for tag, rel in CASES:
    gt_p = E/'out'/f'gt_targets_{tag}'/'render_mask.npy'
    if not gt_p.exists() or not (T2/rel).exists():
        print(f'{tag}: missing ({gt_p.exists()=}, mesh={(T2/rel).exists()})'); continue
    gt = np.load(gt_p); res = gt.shape[0]
    m = trimesh.load(T2/rel, process=False, force='mesh')
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    for flip in (False, True):
        ours = raster(V, F, res, flip)
        iou = (ours & gt).sum()/max((ours | gt).sum(), 1)
        print(f'{tag:20s} res={res} flip_y={str(flip):5s}  IoU_vs_nvdiffrast={iou:.4f}'
              f'  {"<== MATCH" if iou > 0.98 else ""}')
