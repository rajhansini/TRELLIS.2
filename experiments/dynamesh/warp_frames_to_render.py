"""warp_frames_to_render.py — bake the solved rotation into the mesh and warp the
video frames onto it, the way batch G's car clips were fixed.

INPUT  out/orient_similarity/<name>.json from solve_similarity.py.
OUTPUT data/<obj>/mesh/<obj>_render_frame.obj      rotation baked, canonical scale
       data/<obj>/frames_from_video_raw/           the untouched upload
       data/<obj>/frames_from_video/               warped, what training reads

WHY THE FRAMES MOVE AND NOT THE MESH. The whole pipeline downstream -- render_mask,
make_gt_targets, every renderer -- assumes the canonical [-0.5,0.5] mesh and the fixed
camera. Baking a scale into the mesh would put it outside that box and silently change
what every later stage rasterises. The cars were corrected by warping their frames
(data/car_effect1/frames_from_video_raw is that backup), so this follows the same
convention rather than inventing a second one.

THE WARP IS FITTED IN IMAGE SPACE, not converted from the world-space numbers in the
json. Converting would need the projection's depth term to be exactly right for this
mesh; fitting the same two moments -- mask area and centroid -- directly on the 960^2
masks needs nothing but the two masks, and is checked by the IoU it produces.
"""
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent
_argv = list(sys.argv); sys.argv = [sys.argv[0]]
import solve_orientation as SO
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True, help='key in out/orient_similarity')
ap.add_argument('--obj', required=True, help='data/<obj>')
ap.add_argument('--mesh', required=True)
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--min-iou', type=float, default=0.90)
ap.add_argument('--dry-run', action='store_true')
A = ap.parse_args()

T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
DEV = 'cuda'
# One probe file PER PROCESS. A fixed /tmp path is shared by every job that lands on
# the same node, and two of these racing on it produced a half-written PNG and
# "unrecognized data stream contents" -- which looks like a corrupt input, not a
# collision. Named for the job so the failure would be attributable next time.
PROBE = Path(tempfile.gettempdir()) / f'_warp_probe_{os.environ.get("SLURM_JOB_ID", os.getpid())}.png'


def moments(m):
    a = m.sum()
    ys, xs = np.nonzero(m)
    return a, (xs.mean(), ys.mean()) if a else (0.0, 0.0)


def warp(img, s, dx, dy, bg):
    """Scale about the image centre by s, then translate by (dx, dy) pixels."""
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    # PIL's AFFINE maps OUTPUT -> INPUT, so the matrix is the inverse transform.
    inv = (1.0 / s, 0.0, cx - (cx + dx) / s,
           0.0, 1.0 / s, cy - (cy + dy) / s)
    return img.transform((w, h), Image.AFFINE, inv, resample=Image.BICUBIC,
                         fillcolor=tuple(int(v) for v in bg))


def main():
    rec = json.loads((_HERE / 'out/orient_similarity' / f'{A.name}.json').read_text())
    R = np.array(rec['R'])
    mesh = trimesh.load(A.mesh if A.mesh.startswith('/') else T2 / A.mesh,
                        process=False, force='mesh')
    V = SO.canonical(np.asarray(mesh.vertices, dtype=np.float64).copy()) @ R.T
    F = np.asarray(mesh.faces)
    ras = SO.Raster(F)
    ours = ras.mask(V, A.res).cpu().numpy()

    fdir = T2 / f'data/{A.obj}/frames_from_video'
    raw = T2 / f'data/{A.obj}/frames_from_video_raw'
    src = raw if raw.exists() else fdir           # idempotent: re-warp from the raw
    frames = sorted(src.glob('frame_*.png'))
    assert frames, f'no frames in {src}'
    vm0, bg = SO.video_mask(frames[0], A.res, A.tol)
    print(f'[{A.name}] rotation-only IoU before warp: {SO.iou_t(torch.from_numpy(ours).to(DEV), torch.from_numpy(vm0).to(DEV)):.4f}')

    # fit s, dx, dy on frame 1 by matching mask area then centroid, twice
    s, dx, dy = 1.0, 0.0, 0.0
    im0 = Image.open(frames[0]).convert('RGB')
    for _ in range(4):
        w = warp(im0, s, dx, dy, bg)
        w.save(PROBE)
        vm, _ = SO.video_mask(PROBE, A.res, A.tol)
        a_v, c_v = moments(vm); a_o, c_o = moments(ours)
        if a_v == 0:
            break
        k = np.sqrt(a_o / a_v)
        s *= k
        dx = (dx + (c_o[0] - c_v[0])) * k
        dy = (dy + (c_o[1] - c_v[1])) * k
    vmf, _ = SO.video_mask(PROBE, A.res, A.tol)
    iou = SO.iou_t(torch.from_numpy(ours).to(DEV), torch.from_numpy(vmf).to(DEV))
    cov = (ours & vmf).sum() / max(ours.sum(), 1)
    print(f'[{A.name}] warp s={s:.4f} dx={dx:+.1f} dy={dy:+.1f}  ->  IoU {iou:.4f}, '
          f'our silhouette {100*cov:.2f}% covered by the video')
    if iou < A.min_iou:
        print(f'[{A.name}] REFUSING: {iou:.4f} < {A.min_iou}'); sys.exit(1)
    if A.dry_run:
        print(f'[{A.name}] dry run, nothing written'); return

    mp = T2 / f'data/{A.obj}/mesh/{A.obj}_render_frame.obj'
    mp.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=V, faces=F, process=False).export(mp)
    print(f'[{A.name}] wrote {mp}')

    if not raw.exists():
        shutil.move(str(fdir), str(raw))
        fdir.mkdir(parents=True, exist_ok=True)
        # `frames` was resolved under fdir, which the move just emptied. Re-resolve
        # under raw or every read below opens a path that no longer exists -- the
        # move must not invalidate the list it is iterating.
        frames = [raw / f.name for f in frames]
    for f in frames:
        warp(Image.open(f).convert('RGB'), s, dx, dy, bg).save(fdir / f.name)
    print(f'[{A.name}] wrote {len(frames)} warped frames -> {fdir} '
          f'(originals in {raw.name})')
    (T2 / f'data/{A.obj}/warp.json').write_text(json.dumps(
        dict(scale=s, dx=dx, dy=dy, iou=float(iou), source=str(src),
             rotation_from=f'out/orient_similarity/{A.name}.json'), indent=1))


main()
