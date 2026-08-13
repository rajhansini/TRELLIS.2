"""
check_spot_alignment.py — does the canonical camera reproduce the video's framing?

Renders a mesh through the SAME camera the training pipeline uses and compares the
silhouette against frame 1 of the conditioning video. Frame 1 is (near) untextured,
so it is effectively the render that was fed to the video model: if the two
silhouettes agree, the object needs no alignment and the pipeline runs on it by
pointing three paths at data/<obj>/.

Renders two variants, because the pipeline rasterises v_raw (the mesh as supplied),
not the normalised copy:
  raw   — the mesh exactly as it sits on disk
  norm  — after Trellis2TexturingPipeline.preprocess_mesh (centre, uniform scale
          into [-0.5,0.5], axis swap), which is what makes the canonical camera
          object-independent in the first place.

Read-only: renders and measures, writes only PNGs under out/align_check/.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch
import trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--frame', required=True, help='frame 1 of the conditioning video')
ap.add_argument('--res', type=int, default=960, help='match the video resolution')
ap.add_argument('--bg-thresh', type=float, default=0.95,
                help='a pixel is FOREGROUND if min(R,G,B) < this. Matches the tight '
                     'GT mask used by the training script; the leaky (px<0.99).any() '
                     'form was a measured source of wrong conclusions.')
ap.add_argument('--tag', default='align_check')
ARGS = ap.parse_args()

DEVICE = 'cuda'

# ── the pipeline's camera, verbatim ──────────────────────────────────────────
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ret = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def preprocess_mesh_verts(v):
    """Trellis2TexturingPipeline.preprocess_mesh, vertices only (faces are untouched)."""
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    centre = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    v = (v - centre) * scale
    tmp = v[:, 1].copy()
    v[:, 1] = -v[:, 2]
    v[:, 2] = tmp
    return v


def render_silhouette(verts, faces, res):
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    v = torch.from_numpy(np.asarray(verts)).float().to(DEVICE)
    f = torch.from_numpy(np.asarray(faces)).int().to(DEVICE).contiguous()
    proj = intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
    full = (proj @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    rast, _ = dr.rasterize(ctx, clip, f, (res, res))
    return (rast[0, ..., 3] > 0).cpu().numpy()


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(float(union), 1.0)


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    mesh = trimesh.load(ARGS.mesh, process=False, force='mesh')
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    print(f'[MESH] {Path(ARGS.mesh).name}  verts {len(V):,}  faces {len(F):,}')
    print(f'  bbox min {np.round(V.min(0),4)}  max {np.round(V.max(0),4)}')

    img = np.asarray(Image.open(ARGS.frame).convert('RGB'), dtype=np.float32) / 255.0
    if img.shape[0] != ARGS.res:
        print(f'[WARN] frame is {img.shape[0]}px, rendering at {ARGS.res}px')
    gt = img.min(axis=2) < ARGS.bg_thresh
    print(f'\n[VIDEO] frame 1 foreground {int(gt.sum()):,} px '
          f'({100*gt.mean():.2f}% of image)   bbox {bbox_of(gt)}')

    results = {}
    for name, verts in (('raw', V.copy()), ('norm', preprocess_mesh_verts(V.copy()))):
        m = render_silhouette(verts, F, ARGS.res)
        results[name] = m
        vb = verts.min(0), verts.max(0)
        print(f'\n[RENDER {name}] bbox min {np.round(vb[0],4)} max {np.round(vb[1],4)}')
        print(f'  silhouette {int(m.sum()):,} px ({100*m.mean():.2f}%)   bbox {bbox_of(m)}')
        print(f'  IoU vs video frame 1   =  {iou(m, gt):.4f}')

    out = _HERE / 'out' / ARGS.tag
    out.mkdir(parents=True, exist_ok=True)
    for name, m in results.items():
        # red = render only, green = video only, white = agreement
        rgb = np.zeros((*m.shape, 3), np.uint8)
        rgb[m & ~gt] = (220, 60, 60)
        rgb[gt & ~m] = (60, 180, 90)
        rgb[m & gt] = (245, 245, 245)
        Image.fromarray(rgb).save(out / f'overlap_{name}.png')
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f'render_{name}.png')
    Image.fromarray((gt * 255).astype(np.uint8)).save(out / 'video_frame1_mask.png')

    best = max(results, key=lambda k: iou(results[k], gt))
    b = iou(results[best], gt)
    print(f'\n[VERDICT] best variant = {best}   IoU {b:.4f}')
    if b > 0.90:
        print('  -> canonical camera reproduces the video framing. No alignment needed;')
        print('     point --mesh/--gt-dir/--gt-render-dir at this object and run.')
    elif b > 0.70:
        print('  -> close but not aligned. A similarity transform (scale/translate)')
        print('     should close it; solve before generating gt_targets.')
    else:
        print('  -> NOT aligned. The conditioning render used a different camera or')
        print('     a different mesh transform. Do not generate gt_targets from this.')
    print(f'\n  overlap images: {out}')
    print('  red = render only, green = video only, white = both')


if __name__ == '__main__':
    main()
