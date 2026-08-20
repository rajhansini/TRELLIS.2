"""
render_hero.py — the still that goes to Kling.

WHY IT ROTATES THE MESH INSTEAD OF THE CAMERA
  The whole downstream pipeline is built on ONE fixed camera (EXTRINSICS in
  rung27/render_rung27_orbit/SurfaceRenderer). If the Kling still were shot from a
  moved camera, the generated video would be from a view the training camera cannot
  reproduce, and the fitted mesh would never line up with the GT frames.

  So the yaw/pitch is baked into the MESH: <name>_hero.obj is the rotated mesh, and
  rendering it through the standard camera gives the 3/4 elevated view. Train on
  that same rotated mesh and every stage agrees on where the object is.

WHAT MAKES A GOOD KLING INPUT HERE
  - 3/4 view with elevation, so silhouette and surface both read (defaults 35/20)
  - neutral grey material: no baked colour to fight the requested effect
  - soft key + fill + rim: curvature legible, which is what the effect rides on
  - plain white background, matching every existing asset's alpha threshold (245)

Usage:
  python render_hero.py                      # all meshes, *_uv.obj preferred
  python render_hero.py skull sword          # a subset
  python render_hero.py --yaw 45 --pitch 25
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='*')
ap.add_argument('--yaw', type=float, default=35.0, help='degrees around vertical')
ap.add_argument('--pitch', type=float, default=20.0, help='degrees of elevation')
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--outdir', default=str(HERE / 'hero'))
ap.add_argument('--up', default='y', choices=['y', 'z'],
                help="mesh's own up axis. The pipeline camera treats world Z as up "
                     "(EXTRINSICS: cam_y=(0,0,-1)); odedstein meshes are Y-up, so "
                     "they need mapping first or the pitch tilts the wrong way.")
ap.add_argument('--sheet', action='store_true',
                help='render a yaw x pitch contact sheet instead of one still')
A = ap.parse_args()

DEVICE = 'cuda'
_FX = 1.0 / (2.0 * math.tan(math.radians(20.0)))
EXT = torch.tensor([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 2.], [0, 0, 0, 1.]],
                   dtype=torch.float32)
INT = torch.tensor([[_FX, 0, .5], [0, _FX, .5], [0, 0, 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def i2p(i, n, f):
    r = torch.zeros((4, 4), dtype=i.dtype, device=i.device)
    r[0, 0] = 2 * i[0, 0]; r[1, 1] = 2 * i[1, 1]
    r[0, 2] = 2 * i[0, 2] - 1; r[1, 2] = -2 * i[1, 2] + 1
    r[2, 2] = f / (f - n); r[2, 3] = n * f / (n - f); r[3, 2] = 1.
    return r


UP2Z = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # Y-up -> Z-up


def rot(yaw_deg, pitch_deg):
    """Yaw about world Z (the camera's up is -Z), then pitch about world X."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return Rx @ Rz


def fit_unit(v):
    """Centre and scale into the cube the pipeline expects."""
    lo, hi = v.min(0), v.max(0)
    return (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())


def pick(d: Path):
    """Prefer the UV-unwrapped mesh, and the lower-res variant when one exists."""
    uv = sorted(d.glob('*_uv.obj'))
    if not uv:
        return sorted(d.glob('*.obj'))[:1]
    lo = [p for p in uv if any(k in p.stem for k in ('lowres', 'low_resolution', 'closed'))]
    return [lo[0] if lo else uv[0]]


def main():
    import nvdiffrast.torch as dr
    out = Path(A.outdir); out.mkdir(parents=True, exist_ok=True)
    ctx = dr.RasterizeCudaContext()
    proj = (i2p(INT.to(DEVICE), NEAR, FAR) @ EXT.to(DEVICE)).unsqueeze(0)
    # per-mesh axis + yaw, chosen from axis/AXIS_*.png. No pitch: the pipeline
    # camera already sits above the object, and adding pitch about X is what
    # tipped every mesh over on the first attempt.
    # odedstein meshes live in their own object frame -> axis + yaw baked here.
    # EVERY OTHER asset is already fitted to the pipeline camera by
    # fit_mesh_to_video, so rotating it puts the object on its back. Identity.
    # Verified one by one on GPU axis sheets (axis/GPU_*.png). Every odedstein
    # mesh except the hand is C_z2y; the hand alone is B_y2z, and the sword needs
    # a larger yaw because it is edge-on and covers ~2% of frame at 35.
    CFG = {'skull': ('C', 35), 'hand': ('B', 35), 'mushroom': ('C', 35),
           'wingnut': ('C', 35), 'sword': ('E', 70),
           # statues are Z-up in their own frame -> C, verified on GPU axis sheets
           'nefertiti': ('C', 35), 'armadillo': ('C', 35),
           'lionstatue': ('C', 35), 'falconstatue': ('C', 35)}
    AXM = {'B': UP2Z, 'E': np.array([[1,0,0],[0,-1,0],[0,0,-1]], float),
           'C': np.array([[1,0,0],[0,0,-1],[0,1,0]], float),
           'I': np.eye(3)}

    names = A.names or sorted(p.name for p in HERE.iterdir()
                              if p.is_dir() and p.name != 'hero')
    print(f'{"mesh":<26} {"verts":>9} {"faces":>9} {"cover%":>7}  out')
    for name in names:
        d = HERE / name
        for src in pick(d):
            axk, yaw = CFG.get(name, ('I', 0.0))     # unlisted => already camera-fitted
            R = rot(yaw, 0.0) @ AXM[axk]
            m = trimesh.load(src, process=False, force='mesh')
            v = fit_unit(np.asarray(m.vertices, dtype=np.float64) @ R.T)
            f = np.asarray(m.faces, dtype=np.int32)

            # rotated mesh is what training must use, so it is saved alongside
            hero_obj = out / f'{name}_hero.obj'
            trimesh.Trimesh(vertices=v, faces=f,
                            visual=getattr(m, 'visual', None),
                            process=False).export(hero_obj)

            vt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
            ft = torch.tensor(f, dtype=torch.int32, device=DEVICE).contiguous()
            vh = torch.cat([vt[None], torch.ones_like(vt[None][..., :1])], -1)
            clip = torch.bmm(vh, proj.transpose(-1, -2)).contiguous()
            rast, _ = dr.rasterize(ctx, clip, ft, (A.res, A.res))
            mask = (rast[0, ..., 3] > 0)

            # per-face normals -> smooth-ish shading without needing vertex normals
            tri = vt[ft.long()]
            fn = torch.nn.functional.normalize(
                torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
            nrm = fn[(rast[0, ..., 3].long() - 1).clamp(min=0)]

            # key over the shoulder, fill from the opposite side, rim from behind
            def lam(dirv, k):
                L = torch.tensor(dirv, dtype=torch.float32, device=DEVICE)
                L = L / L.norm()
                return (nrm @ L).clamp(min=0) * k
            shade = (0.26 + lam((-0.5, -1.0, 0.75), 0.72)
                          + lam((0.8, -0.6, 0.15), 0.20)
                          + lam((0.0, 1.0, 0.35), 0.16)).clamp(0, 1)

            img = torch.ones(A.res, A.res, 3, device=DEVICE)
            grey = (shade[..., None] * torch.tensor([0.74, 0.745, 0.76], device=DEVICE))
            img = torch.where(mask[..., None], grey, img)
            arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            png = out / f'{name}_hero.png'
            Image.fromarray(arr).save(png)

            cov = 100.0 * float(mask.float().mean())
            print(f'{name + "/" + src.name:<26} {len(v):>9,} {len(f):>9,} '
                  f'{cov:>6.1f}%  {png.name}')
            if cov < 4:
                print(f'   ^ small in frame — consider a tighter crop or larger object')


if __name__ == '__main__':
    main()
