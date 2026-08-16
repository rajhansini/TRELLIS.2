"""
render_align_panels.py — make the alignment auditable by eye.

The mask panels from solve_orientation.py are binary blobs. To actually JUDGE an
alignment you need to see the video and the mesh as images, side by side, in the
same camera. This writes four PNGs per object:

  1  video.png     the real RGB video frame. THIS IS THE GROUND TRUTH. Nothing
                   about it is ours -- it is what the video model produced.
  2  before.png    our mesh, grey-shaded, through the fixed pipeline camera,
                   with NO rotation applied (centre + uniform scale + canonical
                   axis map only). This is where the mesh sat before the solve.
  3  after.png     the same render with the SOLVED rotation applied. This is the
                   pose that would be used to build gt_targets.
  4  overlay.png   silhouette agreement, after the solve:
                     BLUE   both agree            (want: nearly everything)
                     RED    our mesh, no video    (mesh sticks out)
                     GREEN  video, no mesh        (video sticks out)

WHY GREY-SHADED AND NOT A SILHOUETTE
  A silhouette cannot tell a cow facing left from a cow facing right, and it
  cannot show that a teapot's spout is on the wrong side. Lambert shading against
  the camera makes orientation visible, which is the entire thing being judged
  here. IoU still decides pass/fail; this is what lets a human overrule it.

THE CONTROL IS FIRST ON PURPOSE
  spot_lava already trains and scores 0.9913. Its row shows what a CORRECT
  alignment looks like in these exact panels. Every other row should be read
  against it, not against an abstract idea of "close enough".

Rotations are read from out/orient/orientation.json -- this file solves nothing,
it only draws what solve_orientation.py already decided.
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
ap.add_argument('--orient', default='out/orient/orientation.json')
ap.add_argument('--res', type=int, default=640)
ap.add_argument('--frame', type=int, default=1)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--out', default='out/align_panels')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.],
                           [0., 1., 0., 2.], [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5], [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


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
    if n <= 1:
        return m
    return lb == (1 + int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_rgb_and_mask(path, res, tol):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im, dtype=np.int16)
    border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    bg = np.median(border, axis=0)
    m = largest_component(np.abs(a - bg).max(axis=2) > tol)
    rgb = np.asarray(im.resize((res, res), Image.LANCZOS))
    mm = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                    .resize((res, res), Image.NEAREST)) > 127
    return rgb, mm


def shade(V, F, res):
    """Lambert-ish grey render + silhouette, through the fixed camera."""
    import nvdiffrast.torch as dr
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    vn = np.asarray(mesh.vertex_normals)
    ctx = dr.RasterizeCudaContext()
    v = torch.from_numpy(np.ascontiguousarray(V)).float().to(DEVICE)
    f = torch.from_numpy(F).int().to(DEVICE).contiguous()
    n = torch.from_numpy(np.ascontiguousarray(vn)).float().to(DEVICE)
    full = (i2p(INTRINSICS.to(DEVICE), NEAR, FAR) @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    eye = -(EXTRINSICS[:3, :3].T @ EXTRINSICS[:3, 3]).to(DEVICE)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
    rast, _ = dr.rasterize(ctx, torch.bmm(vh, full.transpose(-1, -2)).contiguous(),
                           f, (res, res))
    msk = (rast[0, ..., 3] > 0).cpu().numpy()
    nn = torch.nn.functional.normalize(
        dr.interpolate(n.unsqueeze(0).contiguous(), rast, f)[0][0], dim=-1)
    p = dr.interpolate(v.unsqueeze(0).contiguous(), rast, f)[0][0]
    vd = torch.nn.functional.normalize(eye - p, dim=-1)
    lam = (nn * vd).sum(-1).clamp(0, 1).cpu().numpy()
    img = np.ones((res, res), np.float32)
    img[msk] = 0.18 + 0.72 * lam[msk]          # keep it off pure black and white
    return (img * 255).astype(np.uint8), msk


def main():
    OUT = (_HERE / A.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(_HERE / A.orient))
    print(f'drawing {len(rep)} objects at {A.res}^2 -> {OUT}', flush=True)

    index = []
    for r in rep:
        name = r['object'].split()[0]
        mesh = trimesh.load(T2 / r['mesh'], process=False, force='mesh')
        V0 = canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
        F = np.asarray(mesh.faces)
        R = np.array(r['R'], dtype=np.float64)

        rgb, vmask = video_rgb_and_mask(
            T2 / r['frames'] / f'frame_{A.frame:04d}.png', A.res, A.tol)
        before, _ = shade(V0, F, A.res)
        after, amask = shade(V0 @ R.T, F, A.res)

        ov = np.full((A.res, A.res, 3), 255, np.uint8)
        both = amask & vmask
        ov[both] = (60, 90, 220)                 # blue  — agree
        ov[amask & ~vmask] = (215, 60, 50)       # red   — mesh only
        ov[vmask & ~amask] = (40, 165, 90)       # green — video only

        for tag, arr in (('video', rgb), ('before', before), ('after', after),
                         ('overlay', ov)):
            im = Image.fromarray(arr)
            im.save(OUT / f'{name}_{tag}.png', optimize=True)

        index.append(dict(name=name, iou_before=r['iou_before'], iou_after=r['iou_after'],
                          verdict=r['verdict'], ambiguous=r['ambiguous'],
                          render_px=r['render_px'], video_px=r['video_px'],
                          agree_px=int(both.sum()),
                          mesh_only=int((amask & ~vmask).sum()),
                          video_only=int((vmask & ~amask).sum()),
                          frames=r['frames'], mesh=r['mesh']))
        print(f'  {name:<24} before {r["iou_before"]:.4f}  after {r["iou_after"]:.4f}  '
              f'{r["verdict"]}', flush=True)

    json.dump(index, open(OUT / 'panels.json', 'w'), indent=2)
    print(f'\n[SAVE] {OUT}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
