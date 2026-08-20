"""hero_raster.py — hero stills with TRIANGLE RASTERISATION instead of vertex splatting.

WHY THIS EXISTS
  hero_cpu.py projects each VERTEX to one integer pixel and writes depth there.
  Faces are never filled. Wherever vertices are sparse relative to screen area
  the silhouette has holes: napoleon is 49,501 verts over ~221,000 px, i.e. 4.5 px
  per vertex, so gaps are guaranteed. binary_closing(5x5) patches pinholes, not
  the larger voids at the base of the bust.

  This file fills triangles instead, so vertex density stops mattering. A 1,950-vert
  trumpet renders solid; hero_cpu.py rendered it as 0% coverage (nothing at all).

WHAT IS IDENTICAL TO hero_cpu.py, ON PURPOSE
  RES, the intrinsics fx = 1/(2 tan 20deg), the EXTRINSICS matrix, fit() unit
  normalisation, the axis maps, yawR, the depth->shade curve (0.24 + 0.66*sh with
  sh = 1 - 0.70*normalised_depth) and the slightly cool grey tint. The camera and
  the look are unchanged; only the fill method differs.

  hero_cpu.py IS NOT MODIFIED and the existing hero/*.png are NOT overwritten.
  Output goes to --out (default hero_raster/) so the two can be compared before
  anything is replaced.

METHOD
  Painter's algorithm: backface-cull, sort faces far->near by mean camera depth,
  fill each as a flat polygon shaded by its own depth. No z-buffer needed because
  the draw order resolves occlusion, and flat-per-face shading over a dense mesh
  is visually indistinguishable from the interpolated splat shading at 960px.
"""
import argparse, math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
RES = 960
_FX = 1.0 / (2.0 * math.tan(math.radians(20.0)))
EXT = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 2.], [0, 0, 0, 1.]])
AX = {'none':    np.eye(3),
      'B_y2z':   np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
      'E_flipz': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)}


def yawR(d):
    y = math.radians(d)
    return np.array([[math.cos(y), -math.sin(y), 0],
                     [math.sin(y),  math.cos(y), 0], [0, 0, 1]])


def fit(v):
    lo, hi = v.min(0), v.max(0)
    return (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())


def render(V, F):
    """Filled-triangle render. Returns (rgb float array, coverage percent)."""
    vh = np.concatenate([V, np.ones((len(V), 1))], 1)
    cam = (EXT @ vh.T).T
    z = cam[:, 2].copy()
    z[z < 1e-6] = 1e-6
    x = (_FX * cam[:, 0] / z + 0.5) * RES
    y = (1.0 - (_FX * cam[:, 1] / z + 0.5)) * RES
    P = np.stack([x, y], 1)

    tri = P[F]                                   # (nF, 3, 2) screen-space
    # backface cull by signed area; keeps roughly half the faces and, more
    # importantly, stops interior back-faces being painted over the front.
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    keep = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 0
    if keep.sum() < 0.05 * len(F):               # winding is the other way round
        keep = ~keep
    F2, tri = F[keep], tri[keep]
    if len(F2) == 0:
        return None, 0.0

    fz = z[F2].mean(1)                           # per-face depth
    lo, hi = fz.min(), fz.max()
    sh = 1.0 - 0.70 * ((fz - lo) / max(hi - lo, 1e-6))
    grey = np.clip(0.24 + 0.66 * sh, 0, 1)

    order = np.argsort(-fz)                      # far -> near
    im = Image.new('L', (RES, RES), 255)
    d = ImageDraw.Draw(im)
    for i in order:
        t = tri[i]
        v = int(round(grey[i] * 255))
        d.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])],
                  fill=v, outline=v)             # outline=fill closes seam pixels
    img = np.asarray(im).astype(np.float32) / 255.0
    cover = 100.0 * (img < 0.98).mean()
    return np.dstack([img * 0.98, img * 0.985, img]), cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True, help='path to the .obj/.ply')
    ap.add_argument('--name', required=True, help='output stem, e.g. napoleon')
    ap.add_argument('--axis', default='none', choices=list(AX))
    ap.add_argument('--yaw', type=float, default=0.0)
    ap.add_argument('--out', default=str(HERE / 'hero_raster'))
    ap.add_argument('--write-obj', action='store_true',
                    help='also write <name>_hero.obj with the rotation baked in')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = trimesh.load(a.mesh, process=False, force='mesh')
    V = fit(np.asarray(m.vertices, float) @ (yawR(a.yaw) @ AX[a.axis]).T)
    F = np.asarray(m.faces)
    rgb, cover = render(V, F)
    if rgb is None:
        raise SystemExit(f'{a.name}: nothing rasterised')
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out / f'{a.name}_hero.png')
    if a.write_obj:
        trimesh.Trimesh(vertices=V, faces=F, visual=getattr(m, 'visual', None),
                        process=False).export(out / f'{a.name}_hero.obj')
    print(f'  {a.name:<16} axis={a.axis:<7} yaw={a.yaw:>5.0f}  '
          f'{len(V):>7,}v {len(F):>7,}f  coverage {cover:.1f}%')


if __name__ == '__main__':
    main()
