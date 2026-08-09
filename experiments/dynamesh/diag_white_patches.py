"""
diag_white_patches.py — where do rung17's white patches come from?

Observed: the frozen arm has 0 near-white pixels on the object at any orbit
angle; rung17 has 288-900, worst at orbit frame 45 (yaw 105.6, 4.0% of the
object). The GT targets are not the source — their in-silhouette white is
excluded by the valid mask almost exactly (1001 of 1002 white px at f0001).

The remaining candidate is unsupervised drift: the LoRA only ever sees the
training camera's view, so surface that camera never covered gets gradient
only through shared weights, with nothing checking the answer. This rasterises
the mesh from the ORBIT camera and interpolates each vertex's supervision
coverage, then asks whether the white pixels land on unsupervised surface.

Pure raster + numpy on the already-rendered frames. No pipeline, no sampling.
"""

import argparse, json, math, os, sys
from pathlib import Path

os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', default='/net/projects/ranalab/rajhansini/TRELLIS/'
                                  'render/out/f0075/frozen_f0075.ply')
ap.add_argument('--targets-npz', default=str(_HERE / 'out/gt_backproj/targets.npz'))
ap.add_argument('--orbit-dir', default=str(_HERE / 'out/rung17_360/frames'))
ap.add_argument('--n-orbit', type=int, default=150)
ap.add_argument('--render-res', type=int, default=518)
ap.add_argument('--label-bar', type=int, default=28)
ap.add_argument('--radius', type=float, default=2.0)
ap.add_argument('--elev', type=float, default=0.0)
ap.add_argument('--white-thresh', type=float, default=0.85)
ap.add_argument('--tag', default='white_patch_diag')
args = ap.parse_args()

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image

DEVICE = torch.device('cuda')
OUT = (_HERE / 'out' / args.tag).resolve()
OUT.mkdir(parents=True, exist_ok=True)

NEAR, FAR = 0.5, 3.0
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
INTRINSICS = torch.tensor([[_FX, 0., 0.5], [0., _FX, 0.5], [0., 0., 1.]],
                          dtype=torch.float32)
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)


def log(*a):
    print(*a, flush=True)


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0], r[1, 1] = 2 * fx, 2 * fy
    r[0, 2], r[1, 2] = 2 * cx - 1, -2 * cy + 1
    r[2, 2], r[2, 3] = far / (far - near), near * far / (near - far)
    r[3, 2] = 1.0
    return r


def orbit_extrinsics(yaw_deg, elev_deg, radius):
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = right, -up, fwd
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32)


def main():
    log('=' * 88)
    log('WHITE PATCH DIAGNOSTIC — rung17 360 turntable')
    log('=' * 88)

    d = np.load(args.targets_npz)
    cvx = d['counts_vertex']                       # [F, V] int32
    F, V = cvx.shape
    ever = (cvx > 0).any(axis=0)                   # supervised in >=1 frame
    always = (cvx > 0).all(axis=0)                 # supervised in every frame
    log(f'[SUPERVISION] vertices {V:,}   ever-supervised {ever.sum():,} '
        f'({100*ever.mean():.1f}%)   every-frame {always.sum():,} '
        f'({100*always.mean():.1f}%)   never {int((~ever).sum()):,}')

    mesh = trimesh.load(args.mesh, process=False, force='mesh')
    assert len(mesh.vertices) == V
    v_raw = torch.from_numpy(np.asarray(mesh.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh.faces)).int().to(DEVICE).contiguous()
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    ctx = dr.RasterizeCudaContext()
    R = args.render_res
    proj = intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)

    # per-vertex supervision coverage, interpolated so we can read it per pixel
    cov = torch.from_numpy(ever.astype(np.float32)).to(DEVICE)
    attr = cov.reshape(1, V, 1).contiguous()

    # GATE: the orbit camera at yaw 0 must be the confirmed training view
    gate = float((orbit_extrinsics(0., 0., args.radius) - EXTRINSICS).abs().max())
    log(f'[GATE-cam] |orbit(0,0,{args.radius}) - training EXTRINSICS| = {gate:.3e}')
    assert gate < 1e-5, 'orbit camera != training camera at yaw 0'

    rows = []
    for k in range(1, args.n_orbit + 1):
        p = Path(args.orbit_dir) / f'{k:04d}.png'
        if not p.exists():
            continue
        yaw = (k - 1) * 360.0 / args.n_orbit
        ext = orbit_extrinsics(yaw, args.elev, args.radius).to(DEVICE)
        clip = torch.bmm(vh, (proj @ ext).unsqueeze(0).transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, faces, (R, R))
        vis = rast[0, ..., 3] > 0
        sup = dr.interpolate(attr, rast, faces)[0][0, ..., 0]   # [R,R] in [0,1]
        vis_np, sup_np = vis.cpu().numpy(), sup.cpu().numpy()

        im = np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        W = im.shape[1] // 2
        lo = im[args.label_bar:, W:]                            # rung17 panel
        assert lo.shape[:2] == (R, R), f'panel {lo.shape} != raster {R}'
        obj = vis_np                                            # geometry-defined
        white = (lo.min(axis=2) > args.white_thresh) & obj

        n_obj, n_w = int(obj.sum()), int(white.sum())
        if n_obj == 0:
            continue
        # supervision coverage on white pixels vs on the rest of the object
        rest = obj & ~white
        rows.append(dict(
            k=k, yaw=round(yaw, 1), obj=n_obj, white=n_w,
            white_pct=round(100 * n_w / n_obj, 2),
            sup_white=round(float(sup_np[white].mean()), 4) if n_w else None,
            sup_rest=round(float(sup_np[rest].mean()), 4),
            unsup_frac_white=round(float((sup_np[white] < 0.5).mean()), 4) if n_w else None,
            unsup_frac_rest=round(float((sup_np[rest] < 0.5).mean()), 4),
            bright=round(float(lo[obj].mean()), 4),
        ))
        if k % 15 == 0 or k in (1, 45):
            r = rows[-1]
            log(f"  orbit {k:3d}  yaw {r['yaw']:5.1f}  obj {r['obj']:6,}  "
                f"white {r['white']:5,} ({r['white_pct']:4.1f}%)  "
                f"supcov white {str(r['sup_white']):6}  rest {r['sup_rest']:.4f}")

    log('\n' + '=' * 88)
    wp = np.array([r['white_pct'] for r in rows])
    has_w = [r for r in rows if r['white'] > 0]
    sw = np.array([r['sup_white'] for r in has_w])
    sr = np.array([r['sup_rest'] for r in has_w])
    log(f'[WHITE] frames {len(rows)}  white% min {wp.min():.2f}  '
        f'max {wp.max():.2f}  mean {wp.mean():.2f}')
    log(f'[COVERAGE] mean supervision coverage  ON WHITE {sw.mean():.4f}   '
        f'ON REST {sr.mean():.4f}   ratio {sw.mean()/max(sr.mean(),1e-9):.3f}')
    uw = np.array([r['unsup_frac_white'] for r in has_w])
    ur = np.array([r['unsup_frac_rest'] for r in has_w])
    log(f'[UNSUP]    fraction of pixels on never-supervised surface  '
        f'WHITE {uw.mean():.4f}   REST {ur.mean():.4f}   '
        f'ratio {uw.mean()/max(ur.mean(),1e-9):.2f}x')
    worst = max(rows, key=lambda r: r['white_pct'])
    log(f'[WORST] orbit {worst["k"]} yaw {worst["yaw"]}  '
        f'white {worst["white"]:,}/{worst["obj"]:,} ({worst["white_pct"]}%)  '
        f'supcov white {worst["sup_white"]} vs rest {worst["sup_rest"]}')
    json.dump(rows, open(OUT / 'per_frame.json', 'w'), indent=1)
    log(f'[SAVE] {OUT}/per_frame.json\n[DONE]')


if __name__ == '__main__':
    main()
