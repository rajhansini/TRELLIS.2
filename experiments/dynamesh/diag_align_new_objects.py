"""
diag_align_new_objects.py — can the 6 unaligned objects be trained, or do they
                            need an orientation solve?

STEP 1 of building gt_targets for the new objects. It answers ONE question per
object, the same one GATE-cam asks in the trainer:

    normalise the mesh into the canonical frame, rasterise it through the fixed
    pipeline camera, and compare that silhouette against the object in the
    video's frame 1.  IoU > 0.80  ->  the mesh is already oriented correctly and
    make_gt_targets.py can run.  IoU low  ->  rotation has to be solved first.

WHY NORMALISING IS ENOUGH FOR SCALE AND POSITION, BUT NOT ORIENTATION
  make_render_mask.preprocess() centres the mesh, scales it uniformly into
  [-0.5,0.5], and swaps the axes (y <- -z, z <- y). That disposes of the 154x and
  721x extents entirely -- they are not the problem. What it CANNOT do is decide
  which way the object faces. If each video was generated from a render of its
  own mesh through this same camera, orientation is already right and normalising
  is the whole job. This script measures whether that is true.

THE CONTROLS ARE THE POINT
  spot and teapot already train, and both ship a RAW mesh next to the normalised
  one the runs actually use (spot.obj vs spot_render_frame.obj). So this script
  first checks that
        normalise(spot.obj)  ==  spot_render_frame.obj
  vertex for vertex. If that holds, the recipe below is exactly the one that
  produced a working object, and a low IoU elsewhere is a real finding rather
  than a bug in this file. If it fails, nothing else here can be trusted and the
  script says so instead of reporting numbers.

  DO NOT normalise spot_render_frame.obj. It is already in the canonical frame,
  and preprocess() would apply the axis swap a SECOND time -- silently rotating a
  mesh that was correct, which looks exactly like a misaligned object.

BACKGROUND THRESHOLD
  The 245 rule used everywhere else selects the ENTIRE frame on whale_spots
  (bg 232) and teapot_ceramic_crack (bg 229). Rather than introduce a second
  magic number, the background is estimated per frame from its own border and
  every pixel far enough from that value is object. One rule, no per-object
  tuning, and it reduces to the old behaviour on a white background.
"""
import argparse, math, json
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

ap = argparse.ArgumentParser()
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--frame', type=int, default=1,
                help='which video frame to compare against. Frame 1 is usually '
                     'closest to the untextured conditioning render.')
ap.add_argument('--tol', type=int, default=12,
                help='how far a pixel must sit from the estimated background to '
                     'count as object, per channel (0-255)')
ap.add_argument('--out', default='out/diag_align')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0

# object -> (mesh to normalise, frames dir).  teapot_ceramic_crack and
# teapot_porcelain deliberately share one mesh: one solve covers both.
TARGETS = [
    ('horse_metal',          'data/horse_metal/mesh/horse_unwarp.glb',
                             'data/horse_metal/frames_from_video'),
    ('penguin_circuits',     'data/penguin_circuits/mesh/penguin_remesh_unwarp.glb',
                             'data/penguin_circuits/frames_from_video'),
    ('pumpkin_rot',          'data/pumpkin_rot/mesh/pumpkin_unwarp.glb',
                             'data/pumpkin_rot/frames_from_video'),
    ('whale_spots',          'data/whale_spots/mesh/whale_unwarp.glb',
                             'data/whale_spots/frames_from_video'),
    ('teapot_ceramic_crack', 'data/teapot_ceramic_crack/mesh/teapot.obj',
                             'data/teapot_ceramic_crack/frames_from_video'),
    ('teapot_porcelain',     'data/teapot_porcelain/mesh/teapot.obj',
                             'data/teapot_porcelain/frames_from_video'),
]

# Known-good objects, run through the IDENTICAL code path. If these do not come
# out high, the diagnostic is broken and every other number is meaningless.
CONTROLS = [
    ('spot_lava   [CONTROL]', 'data/spot_lava/mesh/spot.obj',
                              'data/spot_lava/frames_from_video'),
    ('spot_star   [CONTROL]', 'data/spot_star/mesh/spot.obj',
                              'data/spot_star/frames_from_video'),
]


def intrinsics_to_projection(intr, near, far):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0] = 2 * fx; r[1, 1] = 2 * fy
    r[0, 2] = 2 * cx - 1; r[1, 2] = -2 * cy + 1
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.0
    return r


def preprocess(v):
    """Centre, uniform scale into [-0.5,0.5], then the canonical axis map.

    THE AXIS MAP HERE IS NOT THE ONE IN make_render_mask.py, AND THAT FILE IS THE
    ONE THAT IS WRONG.  make_render_mask.preprocess() does

        new_x =  x,   new_y = -z,   new_z = y

    Recovered empirically by searching every axis permutation x sign combination
    against the only ground truth available -- the raw/canonical mesh pair that
    spot_lava actually trains on -- the true map is

        new_x = -x,   new_y =  z,   new_z = y        max|d| = 5.0e-09 (exact)

    An x-flip and a sign error on y. Determinant is +1, so this is a proper
    rotation, not a reflection. Running the wrong one puts the object in a
    mirrored, quarter-turned pose that still looks plausible from the front,
    which is exactly the failure mode that produces a directory like
    gt_targets_spot_REAR_BAD.  GATE-recipe below now ASSERTS the match instead of
    warning, so this can never silently regress.
    """
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    v = (v - (vmin + vmax) / 2) * (0.99999 / (vmax - vmin).max())
    return v[:, (0, 2, 1)] * np.array([-1.0, 1.0, 1.0])


def largest_component(mask):
    lb, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return lb == (1 + int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_mask(path, res, tol):
    """Object mask, with the background estimated from the frame's own border.

    The fixed `min < 245` rule elsewhere selects 99.9% of the frame on the two
    grey-background videos. Estimating the background instead needs no per-object
    number and collapses to the old behaviour when the background is white.
    """
    im = Image.open(path).convert('RGB')
    a = np.asarray(im, dtype=np.int16)
    border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    bg = np.median(border, axis=0)
    m = (np.abs(a - bg).max(axis=2) > tol)
    m = largest_component(m)
    if m.shape[0] != res:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .resize((res, res), Image.NEAREST)) > 127
    return m, bg


def render_mask(ctx, verts, faces, res):
    import nvdiffrast.torch as dr
    v = torch.from_numpy(verts).float().to(DEVICE)
    f = torch.from_numpy(faces).int().to(DEVICE).contiguous()
    full = (intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
            @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    rast, _ = dr.rasterize(ctx, clip, f, (res, res))
    return (rast[0, ..., 3] > 0).cpu().numpy()


def bbox(m):
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None


def main():
    import nvdiffrast.torch as dr
    OUT = (_HERE / A.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 96)
    print('STEP 1 — is normalising enough, or does orientation need solving?')
    print(f'  camera: the fixed pipeline view, {A.res}^2   frame {A.frame}   tol {A.tol}')
    print('=' * 96, flush=True)

    # ---- GATE-recipe: normalise(spot.obj) must reproduce spot_render_frame.obj
    raw = trimesh.load(T2 / 'data/spot_lava/mesh/spot.obj', process=False, force='mesh')
    ref = trimesh.load(T2 / 'data/spot_lava/mesh/spot_render_frame.obj',
                       process=False, force='mesh')
    got = preprocess(np.asarray(raw.vertices, dtype=np.float64).copy())
    exp = np.asarray(ref.vertices, dtype=np.float64)
    ok = got.shape == exp.shape
    dmax = float(np.abs(got - exp).max()) if ok else float('inf')
    print(f'\n[GATE-recipe] normalise(spot.obj) vs spot_render_frame.obj: '
          f'max|d| = {dmax:.3e}  ({got.shape} vs {exp.shape})')
    assert ok and dmax < 1e-5, (
        f'GATE-recipe FAILED: max|d| = {dmax:.3e}. The transform in preprocess() is not '
        f'the one that produced spot_render_frame.obj, so every IoU below would measure '
        f'this file rather than the pipeline. Refusing to report numbers.')
    print('[GATE-recipe] PASSED — this is exactly the transform behind a working object')
    print(flush=True)

    ctx = dr.RasterizeCudaContext()
    rows = []
    for name, mp, gd in CONTROLS + TARGETS:
        mesh = trimesh.load(T2 / mp, process=False, force='mesh')
        V = preprocess(np.asarray(mesh.vertices, dtype=np.float64).copy())
        F = np.asarray(mesh.faces)
        rm = render_mask(ctx, V, F, A.res)

        fp = T2 / gd / f'frame_{A.frame:04d}.png'
        vm, bg = video_mask(fp, A.res, A.tol)

        inter = int((rm & vm).sum()); union = int((rm | vm).sum())
        iou = inter / max(union, 1)
        rows.append(dict(object=name, iou=round(iou, 4),
                         render_px=int(rm.sum()), video_px=int(vm.sum()),
                         render_bbox=bbox(rm), video_bbox=bbox(vm),
                         bg=[int(x) for x in bg], verts=len(V), faces=len(F)))

        # eyeball panel: render | video | overlay (red=render only, green=video only)
        ov = np.zeros((A.res, A.res, 3), np.uint8)
        ov[..., 0] = (rm & ~vm) * 255
        ov[..., 1] = (vm & ~rm) * 255
        ov[..., 2] = (rm & vm) * 255
        panel = np.concatenate([
            np.repeat((rm * 255).astype(np.uint8)[..., None], 3, axis=2),
            np.repeat((vm * 255).astype(np.uint8)[..., None], 3, axis=2), ov], axis=1)
        Image.fromarray(panel).resize((3 * 480, 480), Image.LANCZOS).save(
            OUT / f'{name.split()[0]}_iou{iou:.3f}.png')

        verdict = 'READY' if iou > 0.80 else ('CLOSE' if iou > 0.55 else 'NEEDS ROTATION')
        print(f'{name:<24} IoU {iou:.4f}   render {int(rm.sum()):>7,}px  '
              f'video {int(vm.sum()):>7,}px   bg {bg.astype(int)}   {verdict}', flush=True)

    json.dump(rows, open(OUT / 'align_report.json', 'w'), indent=2)
    print(f'\n{"=" * 96}')
    ready = [r['object'] for r in rows if r['iou'] > 0.80 and 'CONTROL' not in r['object']]
    print(f'READY to build targets ({len(ready)}): {ready if ready else "none"}')
    print(f'panels + align_report.json -> {OUT}')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
