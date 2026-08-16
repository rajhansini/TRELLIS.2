"""
verify_known_rotations.py — check Guan's ground-truth rotations against our camera.

He shipped the exact rotations used to render the Kling input PNGs, re-rendered and
verified byte-identical to the shipped PNGs. That makes them ground truth, not
estimates, and it removes the orientation search entirely.

TWO THINGS THIS DECIDES

1  WHICH FORM.  His matrices live in the raw mesh's frame: y-up, camera on +Z.
   Our pipeline renders canonical(v) = M @ normalize(v) with M = (-x, z, y).
   Writing his recipe in our frame:

       v_his   = R_view @ normalize(R_pre @ v_raw)          (his frame)
       v_ours  = M @ v_his                                   (into our camera)
               = M @ R_total @ normalize(v_raw)
               = (M @ R_total @ M^T) @ canonical(v_raw)

   so the prediction is R' = M @ R_total @ M^T, exactly as his README says.
   Both forms are tested rather than assumed.

   (R_total = R_view @ R_pre is exact here even though normalize sits between
   them: every R_pre in the file is an axis-aligned 90-degree rotation, which
   permutes the bbox extents. normalize divides by max_extent, invariant under a
   permutation, and shifts by bbox_mid, which permutes with it. So
   normalize(R_pre @ v) = R_pre @ normalize(v) identically.)

2  WHICH GATE.  His recommendation, and it is a better gate than the one I used.
   Symmetric IoU punishes video-only surplus, and a cast shadow is exactly that
   -- it is why teapot_ceramic_crack's search chased the shadow and produced a
   worse pose than it started from. COVERAGE asks only "is any of OUR silhouette
   outside the video's object", which a shadow cannot affect, since a shadow only
   ever ADDS to the video mask. Both are reported; coverage is the one to gate on.

Reports his rotation, the conjugated form, and our own solved rotation side by
side, so the convention question is settled by measurement.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

ap = argparse.ArgumentParser()
ap.add_argument('--rotations', required=True, help='kling_rotations.json')
ap.add_argument('--solved', default='out/orient/orientation.json')
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--out', default='out/known_rot')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.],
                           [0., 1., 0., 2.], [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5], [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0
M = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])   # our canonical() axis map (-x, z, y)
F = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])   # his camera frame -> ours (x, -z, y)
# CORRECT FORM: R_ours = F @ R_total @ M.T.  NOT a similarity transform -- our
# canonical map M and the camera frame change F differ by a 180 deg yaw, which is
# why his README's suggested M R M^T lands 174 deg off.  Verified two ways: this
# form sits 5.6-10.8 deg from the four rotations the search solved, exactly the
# range his own note predicts.

# our object  ->  (his key in the json, our mesh, our frames dir)
MAP = [
 ('pumpkin_rot',        'pumpkin', 'data/pumpkin_rot/mesh/pumpkin_unwarp.glb',
                                   'data/pumpkin_rot/frames_from_video'),
 ('horse_metal',        'horse',   'data/horse_metal/mesh/horse_unwarp.glb',
                                   'data/horse_metal/frames_from_video'),
 ('penguin_circuits',   'penguin', 'data/penguin_circuits/mesh/penguin_remesh_unwarp.glb',
                                   'data/penguin_circuits/frames_from_video'),
 ('whale_spots',        'whale',   'data/whale_spots/mesh/whale_unwarp.glb',
                                   'data/whale_spots/frames_from_video'),
 ('teapot_porcelain',   'teapot',  'data/teapot_porcelain/mesh/teapot.obj',
                                   'data/teapot_porcelain/frames_from_video'),
 ('teapot_ceramic_crack','teapot', 'data/teapot_ceramic_crack/mesh/teapot.obj',
                                   'data/teapot_ceramic_crack/frames_from_video'),
]


def i2p(intr, near, far):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0] = 2 * fx; r[1, 1] = 2 * fy
    r[0, 2] = 2 * cx - 1; r[1, 2] = -2 * cy + 1
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.0
    return r


def canonical(v):
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    return v[:, (0, 2, 1)] * np.array([-1.0, 1.0, 1.0])


def largest_component(m):
    lb, n = ndimage.label(m)
    return m if n <= 1 else lb == (1 + int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_mask(path, res, tol):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
    border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    bg = np.median(border, axis=0)
    m = largest_component(np.abs(a - bg).max(axis=2) > tol)
    if m.shape[0] != res:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .resize((res, res), Image.NEAREST)) > 127
    return m


class Raster:
    def __init__(self, faces):
        import nvdiffrast.torch as dr
        self.dr = dr; self.ctx = dr.RasterizeCudaContext()
        self.f = torch.from_numpy(faces).int().to(DEVICE).contiguous()
        self.full = (i2p(INTRINSICS.to(DEVICE), NEAR, FAR) @ EXTRINSICS.to(DEVICE)).unsqueeze(0)

    def mask(self, V, res):
        v = torch.from_numpy(np.ascontiguousarray(V)).float().to(DEVICE)
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, self.full.transpose(-1, -2)).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, self.f, (res, res))
        return (rast[0, ..., 3] > 0).cpu().numpy()


def score(mesh_mask, vid_mask):
    """IoU (shadow-sensitive) and COVERAGE (shadow-immune, the gate to use)."""
    inter = int((mesh_mask & vid_mask).sum())
    union = int((mesh_mask | vid_mask).sum())
    ours = int(mesh_mask.sum())
    uncovered = int((mesh_mask & ~vid_mask).sum())
    return (inter / max(union, 1), 100.0 * uncovered / max(ours, 1), ours, uncovered)


def main():
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    K = json.load(open(A.rotations))['objects']
    try:
        SOLVED = {r['object'].split()[0]: np.array(r['R']) for r in
                  json.load(open(_HERE / A.solved))}
    except Exception:
        SOLVED = {}

    print('=' * 104)
    print('Guan\'s ground-truth rotations vs our camera   (gate: COVERAGE, i.e. % of OUR mesh outside the video)')
    print('=' * 104)
    print(f'{"object":<22}{"variant":<14}{"IoU":>8}{"uncovered":>11}   verdict')
    print('-' * 104)

    rows = []
    for ours, his, mp, gd in MAP:
        if his not in K:
            print(f'{ours:<22}  no entry "{his}" in the json'); continue
        mesh = trimesh.load(T2 / mp, process=False, force='mesh')
        V0 = canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
        FACES = np.asarray(mesh.faces)     # NOT 'F' — that is the frame matrix above
        R = Raster(FACES)
        vm = video_mask(T2 / gd / 'frame_0001.png', A.res, A.tol)

        Rt = np.array(K[his]['R_total'], dtype=np.float64)
        cands = [('guan F R M^T', F @ Rt @ M.T),      # the correct form
                 ('direct', Rt), ('conjugated M R M^T', M @ Rt @ M.T)]
        if ours in SOLVED:
            cands.append(('our solve', SOLVED[ours]))

        best = None
        for name, Rm in cands:
            mm = R.mask(V0 @ Rm.T, A.res)
            iou, unc, px, upx = score(mm, vm)
            ok = 'PASS' if unc < 3.0 else ('marginal' if unc < 8.0 else 'FAIL')
            print(f'{ours:<22}{name:<14}{iou:>8.4f}{unc:>10.2f}%   {ok}  ({px:,} px, {upx:,} outside)')
            rows.append(dict(object=ours, his_key=his, variant=name, iou=round(iou, 4),
                             uncovered_pct=round(unc, 3), mesh_px=px, uncovered_px=upx,
                             verdict=ok, R=[[float(x) for x in r] for r in Rm]))
            if best is None or unc < best[1]:
                best = (name, unc, Rm, mm)
        # panel for the winner
        nm, _, _, mm = best
        ov = np.full((A.res, A.res, 3), 255, np.uint8)
        ov[mm & vm] = (60, 90, 220); ov[mm & ~vm] = (215, 60, 50); ov[vm & ~mm] = (40, 165, 90)
        Image.fromarray(ov).resize((520, 520), Image.LANCZOS).save(OUT / f'{ours}_{nm.replace(" ","")}.png')
        print(f'{"":<22}-> best: {nm}\n' + '-' * 104)

    json.dump(rows, open(OUT / 'known_rot_report.json', 'w'), indent=2)

    # which form won, overall
    by = {}
    for r in rows:
        by.setdefault(r['variant'], []).append(r['uncovered_pct'])
    print('\nmean % of our mesh left uncovered, by variant:')
    for k, v in sorted(by.items(), key=lambda kv: np.mean(kv[1])):
        print(f'  {k:<14} {np.mean(v):.2f}%   (n={len(v)})')
    print(f'\n[SAVE] {OUT}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
