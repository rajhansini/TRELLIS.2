"""
render_backproj_gt.py — turn the backprojected GT into 150 clean 2D targets

The video's silhouette is not ours: it grows 102,549 -> 109,583 px over the
sequence while our render is pinned at 27,900 px because the mesh and camera are.
So a pixel loss against the raw frame is partly a shape comparison.

backproject_gt.py already lifted each frame's colours onto the mesh vertices.
This renders those vertex colours back out through the SAME mesh and the SAME
camera the training loop uses, so the resulting target image has our silhouette
by construction. Training then compares two renders of one mesh — the shape
question is gone and the loss is purely texture, still in 2D.

Vertices the camera never saw carry no target (173,951 of 215,462). They are not
rasterised from this view either, so they contribute nothing here; a per-pixel
validity weight is carried through the interpolation and written alongside, so
the training loss can ignore any sliver where a covered pixel's triangle had no
supervised vertex.
"""

import argparse, json, math, os, sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', default='/net/projects/ranalab/rajhansini/TRELLIS/'
                                  'render/out/f0075/frozen_f0075.ply')
ap.add_argument('--targets-npz', default=str(_HERE / 'out/gt_backproj/targets.npz'))
ap.add_argument('--render-res', type=int, default=518)
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--tag', default='gt_rendered')
ap.add_argument('--min-support', type=float, default=1e-3,
                help='keep a pixel if this much barycentric weight came from '
                     'supervised vertices (was effectively 0.99 = all 3 corners)')
args = ap.parse_args()
MIN_SUPPORT = args.min_support

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image

DEVICE = torch.device('cuda')
OUT = (_HERE / 'out' / args.tag).resolve()
(OUT / 'frames').mkdir(parents=True, exist_ok=True)

NEAR, FAR = 0.5, 3.0
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX, 0., 0.5], [0., _FX, 0.5], [0., 0., 1.]],
                          dtype=torch.float32)


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


def main():
    log('=' * 88)
    log('RENDER BACKPROJECTED GT -> 150 clean 2D targets')
    log('=' * 88)

    d = np.load(args.targets_npz)
    tv = d['targets_vertex']              # [F, V, 3] float16
    cvx = d['counts_vertex']              # [F, V]   int32
    F, V = tv.shape[0], tv.shape[1]
    log(f'[TARGETS] {args.targets_npz}')
    log(f'  targets_vertex {tv.shape}   vertices with a target every frame '
        f'{int(d["valid_vertex"].sum()):,}/{V:,}')

    mesh = trimesh.load(args.mesh, process=False, force='mesh')
    assert len(mesh.vertices) == V, f'mesh has {len(mesh.vertices)} verts, targets {V}'
    v_raw = torch.from_numpy(np.asarray(mesh.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh.faces)).int().to(DEVICE).contiguous()

    R = args.render_res
    full = (intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
            @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, clip, faces, (R, R))
    mask = rast[0, ..., 3] > 0
    log(f'[RASTER] {R}x{R}, silhouette {int(mask.sum()):,} px — identical to the '
        f'training render by construction')

    valid_px = np.zeros(F, np.int64)
    old_px = np.zeros(F, np.int64)
    for f in range(min(F, args.n_frames)):
        col = torch.from_numpy(tv[f].astype(np.float32)).to(DEVICE)          # [V,3]
        wgt = torch.from_numpy((cvx[f] > 0).astype(np.float32)).to(DEVICE)   # [V]
        # PREMULTIPLY. Unsupervised vertices are not zero — they carry up to
        # 0.075 of leftover junk — so their colour has to be killed explicitly
        # before it can be barycentrically blended into a neighbour's pixel.
        attr = torch.cat([col * wgt.unsqueeze(-1), wgt.unsqueeze(-1)], -1).unsqueeze(0)
        out = dr.interpolate(attr.contiguous(), rast, faces)[0][0]           # [R,R,4]
        num, den = out[..., :3], out[..., 3]
        # Normalise rather than threshold. num/den is the barycentric mean over
        # ONLY the supervised corners, so a triangle with 2 good corners and 1
        # unseen one still yields the right colour instead of being discarded.
        # The old code demanded den>0.99 (all three corners) and painted the
        # rest white, which is what speckled the lid rim, shoulder and spout.
        rgb = num / den.clamp_min(1e-6).unsqueeze(-1)
        ok = mask & (den > MIN_SUPPORT)
        img = torch.ones(R, R, 3, device=DEVICE)
        img[ok] = rgb[ok].clamp(0, 1)
        Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
            OUT / 'frames' / f'gt_{f+1:04d}.png')
        np.save(OUT / 'frames' / f'valid_{f+1:04d}.npy', ok.cpu().numpy())
        valid_px[f] = int(ok.sum())
        old_px[f] = int((mask & (den > 0.99)).sum())      # what the old rule kept
        if (f + 1) % 25 == 0 or f == 0:
            log(f'  {f+1:3d}/{min(F, args.n_frames)}  usable px {valid_px[f]:,}'
                f'   (old all-3-corners rule: {old_px[f]:,}, '
                f'recovered {valid_px[f]-old_px[f]:+,})')

    log(f'\n[USABLE] per-frame target pixels: min {valid_px[:args.n_frames].min():,}  '
        f'max {valid_px[:args.n_frames].max():,}  '
        f'mean {int(valid_px[:args.n_frames].mean()):,}')
    log(f'         render silhouette {int(mask.sum()):,} px — the shortfall is the '
        f'sliver whose triangles had NO supervised corner at all')
    _o, _n = old_px[:args.n_frames], valid_px[:args.n_frames]
    log(f'[FRINGE] white px inside the silhouette:  old rule {int(mask.sum())-_o.mean():,.0f} '
        f'-> new rule {int(mask.sum())-_n.mean():,.0f}   '
        f'({100*(1-(int(mask.sum())-_n.mean())/max(int(mask.sum())-_o.mean(),1)):.1f}% of the '
        f'fringe recovered)')
    np.save(OUT / 'render_mask.npy', mask.cpu().numpy())
    json.dump(dict(mesh=args.mesh, render_res=R, n_frames=int(args.n_frames),
                   silhouette_px=int(mask.sum()),
                   usable_px_min=int(valid_px[:args.n_frames].min()),
                   usable_px_max=int(valid_px[:args.n_frames].max())),
              open(OUT / 'gt_rendered.json', 'w'), indent=2)
    log(f'[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
