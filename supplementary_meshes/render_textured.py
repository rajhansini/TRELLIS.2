"""render_textured.py -- the exposed-area front view, rendered WITH the mesh's own texture.

best_view.py renders grey because the Kling pipeline's round-1 input is a grey mesh. A
textured render is what round 2 (continuation) needs: Kling is handed a surface that
already carries texture and asked to keep it moving.

Same camera as best_view.py and guanc/tmp/kling/render_final.py -- res 1024, dist 2.6,
fov 30, same two-light rig -- so the output is drop-in for either pipeline.

Colour comes from the mesh's own UVs: points are sampled on the faces AREA-WEIGHTED and
their barycentric coordinates carry the UV lookup, so the texture is sampled where the
surface actually is rather than being reprojected.
"""
import sys, math, json, pathlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from best_view import UPS, normalize, basis, project, RES, DIST, FOV   # one camera, one source

HERE = pathlib.Path(__file__).resolve().parent
NRENDER = 3_000_000


def load_obj_uv(p):
    """-> V (nv,3), VT (nt,2), F (nf,3) vertex idx, FT (nf,3) uv idx"""
    V, VT, F, FT = [], [], [], []
    for ln in open(p, errors='ignore'):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('vt '):
            VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith('f '):
            t = ln.split()[1:]
            vi = [int(x.split('/')[0]) - 1 for x in t]
            ti = [int(x.split('/')[1]) - 1 if len(x.split('/')) > 1 and x.split('/')[1] else 0
                  for x in t]
            for k in range(1, len(vi) - 1):
                F.append([vi[0], vi[k], vi[k + 1]]); FT.append([ti[0], ti[k], ti[k + 1]])
    return (np.asarray(V, np.float64), np.asarray(VT, np.float64),
            np.asarray(F, np.int64), np.asarray(FT, np.int64))


def sample_uv(V, VT, F, FT, n, seed=0):
    """area-weighted surface samples, each carrying its interpolated UV and normal"""
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    keep = area > 0
    idx = np.flatnonzero(keep)
    a, b, c, cr, area = a[keep], b[keep], c[keep], cr[keep], area[keep]
    nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
    i = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    fl = (u + v) > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    P = a[i] + u * (b[i] - a[i]) + v * (c[i] - a[i])
    ft = FT[idx][i]
    ta, tb, tc = VT[ft[:, 0]], VT[ft[:, 1]], VT[ft[:, 2]]
    UV = ta + u * (tb - ta) + v * (tc - ta)      # same barycentric weights as the point
    return P, nrm[i], UV


def render_tex(P, N, UV, tex, yaw, pitch, res=RES):
    eye, M, fwd, right, up = basis(yaw, pitch)
    L1 = -0.4 * right + 0.5 * up - 0.8 * fwd; L1 /= np.linalg.norm(L1)
    L2 = 0.7 * right + 0.2 * up - 0.5 * fwd;  L2 /= np.linalg.norm(L2)
    x, y, z = project(P, eye, M, res)
    face = (N @ fwd) < 0
    ok = face & (x >= 0) & (x < res) & (y >= 0) & (y < res)
    xi, yi, zi, ni, uvi = x[ok].astype(int), y[ok].astype(int), z[ok], N[ok], UV[ok]
    flat = yi * res + xi
    o = np.argsort(zi, kind='stable')
    fs = flat[o]
    _, first = np.unique(fs, return_index=True)
    idx = o[first]
    H, W = tex.shape[:2]
    tu = np.clip((uvi[idx, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
    tv = np.clip((1.0 - (uvi[idx, 1] % 1.0)) * (H - 1), 0, H - 1).astype(int)   # v flipped
    alb = tex[tv, tu].astype(np.float64) / 255.0
    lam = 0.30 + 0.55 * np.clip(ni[idx] @ L1, 0, 1) + 0.25 * np.clip(ni[idx] @ L2, 0, 1)
    img = np.ones((res * res, 3))
    img[fs[first]] = np.clip(alb * lam[:, None], 0, 1)
    hit = np.zeros(res * res, bool); hit[fs[first]] = True
    img, hit = img.reshape(res, res, 3), hit.reshape(res, res)
    solid = binary_closing(hit, np.ones((3, 3)))
    fill = solid & ~hit
    if fill.any():                                   # seal splat pinholes with a local mean
        img[fill] = img[hit].mean(0)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    name, upname, yaw, pitch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    objp = HERE / 'OBJ' / f'{name}.obj'
    texp = pathlib.Path(sys.argv[5])
    V, VT, F, FT = load_obj_uv(objp)
    tex = np.asarray(Image.open(texp).convert('RGB'))
    V = normalize(V) @ UPS[upname].T
    P, N, UV = sample_uv(V, VT, F, FT, NRENDER, seed=1)
    out = HERE / 'front' / f'{name}_textured.png'
    Image.fromarray(render_tex(P, N, UV, tex, yaw, pitch)).save(out)
    print(f'{name}: up={upname} yaw={yaw} pitch={pitch} tex={tex.shape[:2]} -> {out}')
