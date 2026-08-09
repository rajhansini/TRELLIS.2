"""
render_ply_unlit.py — show a .ply's vertex colours with NO lighting

MeshLab shades with a default material and a light, and its colour mode was on
'User-Def', so what it draws is a uniform grey lit surface, not the vertex
colours. Specular highlights then land on rims, spout edges and the lid crease —
which reads as white patches that are not in the data. Measured on this mesh:
max vertex luminance 0.945 (nothing reaches white), and the brightest vertices
show curvature ratio 1.0x against the rest, i.e. they are NOT on edges.

This renders the same mesh with flat barycentric interpolation of the vertex
colours and no shading term at all, so every pixel is the texture and nothing
else. Two panels per angle: colours as-is, and a saturation map so you can see
where colour actually lives.
"""

import argparse, math, os, sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--ply', required=True)
ap.add_argument('--res', type=int, default=800)
ap.add_argument('--radius', type=float, default=2.0)
ap.add_argument('--views', default='0,0:0,55:90,30:180,0',
                help="yaw,elev pairs separated by ':'  (55 elev ~ the MeshLab "
                     "top-down view)")
ap.add_argument('--out', default=None)
A = ap.parse_args()

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image

DEV = torch.device('cuda')
NEAR, FAR = 0.5, 6.0
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))


def proj_from(fx, near, far):
    r = torch.zeros((4, 4), dtype=torch.float32, device=DEV)
    r[0, 0] = r[1, 1] = 2 * fx
    r[2, 2], r[2, 3] = far / (far - near), near * far / (near - far)
    r[3, 2] = 1.0
    return r


def orbit(yaw, elev, radius):
    y, e = math.radians(yaw), math.radians(elev)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)])
    f = -eye / np.linalg.norm(eye)
    up = np.array([0., 0., 1.])
    r = np.cross(f, up)
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1., 0., 0.])
    u = np.cross(r, f)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = r, -u, f
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32, device=DEV)


def main():
    m = trimesh.load(A.ply, process=False, force='mesh')
    v = np.asarray(m.vertices, dtype=np.float32)
    v = v - v.mean(0)                                # centre for a clean orbit
    scale = float(np.abs(v).max())
    v = v / (scale * 2.2)
    col = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
    print(f'[MESH] {len(v):,} verts  {len(m.faces):,} faces')
    print(f'[COLOUR] mean {col.mean():.4f}  max-luminance {col.min(1).max():.4f}  '
          f'verts with all channels >0.95: {int((col.min(1) > 0.95).sum()):,}')

    vt = torch.from_numpy(v).to(DEV)
    ft = torch.from_numpy(np.asarray(m.faces)).int().to(DEV).contiguous()
    ct = torch.from_numpy(col).to(DEV)
    sat = (ct.max(1).values - ct.min(1).values)      # 0 = grey, 1 = pure colour
    ctx = dr.RasterizeCudaContext()
    P = proj_from(_FX, NEAR, FAR)
    R = A.res
    vh = torch.cat([vt, torch.ones_like(vt[:, :1])], -1).unsqueeze(0)

    tiles = []
    for spec in A.views.split(':'):
        yaw, elev = (float(x) for x in spec.split(','))
        clip = torch.bmm(vh, (P @ orbit(yaw, elev, A.radius)).unsqueeze(0)
                         .transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, ft, (R, R))
        msk = rast[0, ..., 3] > 0
        rgb = dr.interpolate(ct.unsqueeze(0).contiguous(), rast, ft)[0][0]
        s = dr.interpolate(sal := sat.reshape(1, -1, 1).contiguous(), rast, ft)[0][0, ..., 0]
        # NO shading term anywhere: what you see is exactly the vertex colour
        img = torch.ones(R, R, 3, device=DEV)
        img[msk] = rgb[msk].clamp(0, 1)
        heat = torch.ones(R, R, 3, device=DEV)
        heat[msk] = torch.stack([s[msk], s[msk] * 0.35,
                                 1.0 - s[msk]], -1).clamp(0, 1)
        n_white = int(((rgb.min(-1).values > 0.95) & msk).sum())
        print(f'  yaw {yaw:6.1f} elev {elev:5.1f}   px {int(msk.sum()):7,}   '
              f'mean {float(rgb[msk].mean()):.3f}   px>0.95 on the object: {n_white}')
        tiles.append(np.concatenate(
            [(img.cpu().numpy() * 255).astype(np.uint8),
             (heat.cpu().numpy() * 255).astype(np.uint8)], axis=0))

    out = A.out or str(Path(A.ply).with_suffix('')) + '_unlit.png'
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
    print(f'[SAVE] {out}   (top row = vertex colours unlit, '
          f'bottom row = saturation: blue grey -> orange saturated)')


if __name__ == '__main__':
    main()
