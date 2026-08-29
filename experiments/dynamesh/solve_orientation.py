"""
solve_orientation.py — find the rotation that puts each mesh where its video is.

STEP 2. diag_align_new_objects.py established that centring + uniform scale +
the canonical axis map (-x, z, y) is exact -- spot_lava scores IoU 0.9913 and its
render_mask matches the one on disk bit for bit -- and that all six new objects
still land wrong. The only remaining unknown is a rotation, so this searches for
it directly.

SEARCH
  coarse   the 24 proper rotations of the cube (axis permutations with det +1).
           Most "wrong way round" meshes are exactly one of these, since they come
           from an exporter's axis convention rather than an arbitrary pose.
  refine   coordinate descent on yaw/pitch/roll around the best few, +-16 deg at
           4 deg then +-4 deg at 1 deg. Cheap: the objective is one rasterisation.

  Scored at --search-res (default 384) because silhouette IoU is stable under
  downsampling and the search is ~1,500 rasterisations per object. The WINNER is
  re-scored at --res (960), which is the number that decides pass/fail, so the
  reported figure is always the full-resolution one.

WHY IoU AND NOT SHADING
  find_orientation.py argued silhouette cannot separate front from back on a
  near-symmetric object, and that is true -- a cow's outline is nearly the same
  either way. This script therefore reports the runner-up alongside the winner:
  when two orientations score within --ambiguous of each other the object is
  flagged AMBIGUOUS and the contact sheet has to be eyeballed before its targets
  are built. Silently picking one of two equal scores is how a mesh ends up
  trained back-to-front.

  (find_orientation.py also carries the WRONG axis map, copied from
  make_render_mask.py -- (x,-z,y) instead of (-x,z,y). Not used here.)

GATE-solve
  spot_lava is searched too, as a control. It is already aligned, so the search
  MUST return essentially identity and IoU ~0.99. If it does not, the search is
  broken and no other result is worth reading.
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

ap = argparse.ArgumentParser()
ap.add_argument('--res', type=int, default=960, help='resolution the winner is scored at')
ap.add_argument('--search-res', type=int, default=384, help='resolution used during the search')
ap.add_argument('--frame', type=int, default=1)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--pass-iou', type=float, default=0.80, help="the trainer's GATE-cam threshold")
ap.add_argument('--ambiguous', type=float, default=0.02,
                help='if the runner-up is within this IoU of the winner, flag it')
ap.add_argument('--out', default='out/orient')
ap.add_argument('--jobs-file', default=None,
                help='JSON list of [name, mesh, frames_dir]; REPLACES the JOBS list '
                     'below. Added for batch D rather than editing that list, which is '
                     'the record of the batch A-C solve and is referenced downstream.')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.],
                           [0., 1., 0., 2.], [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5], [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0

JOBS = [
    ('spot_lava [CONTROL]',  'data/spot_lava/mesh/spot.obj',
                             'data/spot_lava/frames_from_video'),
    ('horse_metal',          'data/horse_metal/mesh/horse_unwarp.glb',
                             'data/horse_metal/frames_from_video'),
    ('penguin_circuits',     'data/penguin_circuits/mesh/penguin_remesh_unwarp.glb',
                             'data/penguin_circuits/frames_from_video'),
    ('pumpkin_rot',          'data/pumpkin_rot/mesh/pumpkin_unwarp.glb',
                             'data/pumpkin_rot/frames_from_video'),
    ('whale_spots',          'data/whale_spots/mesh/whale_unwarp.glb',
                             'data/whale_spots/frames_from_video'),
    # same mesh, two different videos. Solved separately on purpose: if the two
    # agree that is a free consistency check, and if they disagree that is a real
    # finding about how the videos were generated.
    ('teapot_ceramic_crack', 'data/teapot_ceramic_crack/mesh/teapot.obj',
                             'data/teapot_ceramic_crack/frames_from_video'),
    ('teapot_porcelain',     'data/teapot_porcelain/mesh/teapot.obj',
                             'data/teapot_porcelain/frames_from_video'),
]


if A.jobs_file:
    JOBS = [tuple(j) for j in json.loads(open(A.jobs_file).read())]


def intrinsics_to_projection(intr, near, far):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0] = 2 * fx; r[1, 1] = 2 * fy
    r[0, 2] = 2 * cx - 1; r[1, 2] = -2 * cy + 1
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.0
    return r


def canonical(v):
    """centre, uniform scale into [-0.5,0.5], canonical axis map (-x, z, y).
    Verified exact against the spot raw/canonical pair, max|d| = 5.0e-09."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    return v[:, (0, 2, 1)] * np.array([-1.0, 1.0, 1.0])


def cube_rotations():
    """The 24 proper rotations of the cube, as integer matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                out.append(M)
    return out


def euler(yaw, pitch, roll):
    """intrinsic z-y-x, degrees. z is the camera's up after the axis map."""
    a, b, c = map(math.radians, (yaw, pitch, roll))
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1., 0], [-math.sin(b), 0, math.cos(b)]])
    Rx = np.array([[1., 0, 0], [0, math.cos(c), -math.sin(c)], [0, math.sin(c), math.cos(c)]])
    return Rz @ Ry @ Rx


def largest_component(mask):
    lb, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return lb == (1 + int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_mask(path, res, tol):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
    border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    bg = np.median(border, axis=0)
    m = largest_component(np.abs(a - bg).max(axis=2) > tol)
    if m.shape[0] != res:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .resize((res, res), Image.NEAREST)) > 127
    return m, bg


class Raster:
    """Faces and the projection are constant; only the vertices move."""
    def __init__(self, faces):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()
        self.f = torch.from_numpy(faces).int().to(DEVICE).contiguous()
        self.full = (intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
                     @ EXTRINSICS.to(DEVICE)).unsqueeze(0)

    def mask(self, V, res):
        v = torch.from_numpy(np.ascontiguousarray(V)).float().to(DEVICE)
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, self.full.transpose(-1, -2)).contiguous()
        rast, _ = self.dr.rasterize(self.ctx, clip, self.f, (res, res))
        return (rast[0, ..., 3] > 0)


def iou_t(a, b):
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / max(union, 1)


def main():
    OUT = (_HERE / A.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 96)
    print('STEP 2 — solve the rotation that puts each mesh where its video is')
    print(f'  search {A.search_res}^2, winner scored at {A.res}^2, '
          f'pass IoU > {A.pass_iou}')
    print('=' * 96, flush=True)

    CUBE = cube_rotations()
    assert len(CUBE) == 24, f'expected 24 cube rotations, got {len(CUBE)}'
    report = []

    for name, mp, gd in JOBS:
        mesh = trimesh.load(T2 / mp, process=False, force='mesh')
        V0 = canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
        F = np.asarray(mesh.faces)
        R = Raster(F)

        vm_s, bg = video_mask(T2 / gd / f'frame_{A.frame:04d}.png', A.search_res, A.tol)
        vm_hi, _ = video_mask(T2 / gd / f'frame_{A.frame:04d}.png', A.res, A.tol)
        tv_s = torch.from_numpy(vm_s).to(DEVICE)
        tv_hi = torch.from_numpy(vm_hi).to(DEVICE)

        score = lambda M: iou_t(R.mask(V0 @ M.T, A.search_res), tv_s)

        # ---- coarse: the 24 cube rotations
        coarse = sorted(((score(M), i) for i, M in enumerate(CUBE)), reverse=True)
        best_i = coarse[0][1]
        runner = coarse[1][0]

        # ---- refine: coordinate descent on yaw/pitch/roll around the top 3
        best_s, best_M = -1.0, None
        for s0, i in coarse[:3]:
            M = CUBE[i].copy()
            cur = s0
            for step, rng in ((4.0, 16.0), (1.0, 4.0)):
                improved = True
                while improved:
                    improved = False
                    for axis in range(3):
                        for d in np.arange(-rng, rng + 1e-9, step):
                            if d == 0:
                                continue
                            e = [0.0, 0.0, 0.0]; e[axis] = float(d)
                            cand = euler(*e) @ M
                            sc = score(cand)
                            if sc > cur + 1e-6:
                                cur, M, improved = sc, cand, True
            if cur > best_s:
                best_s, best_M = cur, M

        # ---- the number that counts, at full resolution
        m_hi = R.mask(V0 @ best_M.T, A.res)
        iou_hi = iou_t(m_hi, tv_hi)
        base_hi = iou_t(R.mask(V0, A.res), tv_hi)          # before any rotation

        ambiguous = (best_s - runner) < A.ambiguous
        verdict = ('PASS' if iou_hi > A.pass_iou else 'FAIL')
        if ambiguous and verdict == 'PASS':
            verdict = 'PASS (AMBIGUOUS — eyeball it)'

        report.append(dict(object=name, iou_before=round(base_hi, 4),
                           iou_after=round(iou_hi, 4), coarse_best=round(coarse[0][0], 4),
                           coarse_runner_up=round(runner, 4), ambiguous=bool(ambiguous),
                           verdict=verdict, R=[[round(x, 8) for x in r] for r in best_M],
                           render_px=int(m_hi.sum().item()), video_px=int(tv_hi.sum().item()),
                           mesh=str(mp), frames=str(gd), bg=[int(x) for x in bg]))

        mh = m_hi.cpu().numpy()
        ov = np.zeros((A.res, A.res, 3), np.uint8)
        ov[..., 0] = (mh & ~vm_hi) * 255
        ov[..., 1] = (vm_hi & ~mh) * 255
        ov[..., 2] = (mh & vm_hi) * 255
        panel = np.concatenate([
            np.repeat((mh * 255).astype(np.uint8)[..., None], 3, axis=2),
            np.repeat((vm_hi * 255).astype(np.uint8)[..., None], 3, axis=2), ov], axis=1)
        Image.fromarray(panel).resize((3 * 460, 460), Image.LANCZOS).save(
            OUT / f'{name.split()[0]}_{iou_hi:.3f}.png')

        print(f'{name:<24} {base_hi:.4f} -> {iou_hi:.4f}   '
              f'(coarse {coarse[0][0]:.3f}, runner-up {runner:.3f})   {verdict}', flush=True)

    json.dump(report, open(OUT / 'orientation.json', 'w'), indent=2)

    # ---- GATE-solve: the control must come back essentially unchanged
    ctl = report[0]
    print(f'\n[GATE-solve] control {ctl["object"]}: '
          f'{ctl["iou_before"]:.4f} -> {ctl["iou_after"]:.4f}')
    assert ctl['iou_after'] > 0.97, (
        f'GATE-solve FAILED: the control ended at {ctl["iou_after"]:.4f}. spot_lava is '
        f'already aligned, so the search must return it essentially unchanged. '
        f'Every other row is unreliable.')
    print('[GATE-solve] PASSED — the search does not damage an already-correct mesh')

    ok = [r['object'] for r in report[1:] if r['iou_after'] > A.pass_iou]
    amb = [r['object'] for r in report[1:] if r['ambiguous'] and r['iou_after'] > A.pass_iou]
    print(f'\n{"=" * 96}')
    print(f'SOLVED ({len(ok)}/6): {ok if ok else "none"}')
    if amb:
        print(f'AMBIGUOUS, eyeball before building targets: {amb}')
    print(f'panels + orientation.json -> {OUT}')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
