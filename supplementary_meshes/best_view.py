"""best_view.py -- pick, per mesh, the camera that EXPOSES THE MOST SURFACE, and
render it through the Kling conditioning camera.

Why exposed-area and not silhouette area: the video model can only invent texture
for surface it can see. Silhouette area rewards a broadside view of a flat object;
what we want is the fraction of the mesh's own area that is unoccluded and
front-facing, which is what the driving video actually gets to texture.

Scoring: sample points on the faces AREA-WEIGHTED, project them through the real
camera, z-buffer them, and count the survivors. Because the samples are
area-weighted, survivors/total IS the visible area fraction, and it handles
self-occlusion (elk antlers, cello neck) that a normals-only estimate misses.

Camera is copied from guanc/tmp/kling/render_final.py -- res 1024, dist 2.6,
fov 30, same light rig -- so these PNGs are drop-in for the Kling pipeline.
Meshes arrive in arbitrary frames, so the search also picks which axis is up.
"""
import sys, math, json, pathlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing

HERE = pathlib.Path(__file__).resolve().parent
OBJ, OUT = HERE / 'OBJ', HERE / 'front'
OUT.mkdir(exist_ok=True)

RES, DIST, FOV = 1024, 2.6, 30.0
SCORE_RES, NSAMP, NRENDER = 256, 40_000, 1_500_000

# ---- the six ways an axis can be "up" -----------------------------------------
def _m(cols):
    return np.array(cols, float).T
UPS = {                      # name -> 3x3 taking mesh coords to Y-up camera frame
    '+Y': np.eye(3),
    '-Y': _m([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    '+Z': _m([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    '-Z': _m([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    '+X': _m([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    '-X': _m([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}

def load_obj(p):
    V, F = [], []
    for ln in open(p, errors='ignore'):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)

def normalize(V):
    lo, hi = V.min(0), V.max(0)
    return (V - (lo + hi) / 2) / (hi - lo).max()

def sample(V, F, n, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    keep = area > 0
    a, b, c, cr, area = a[keep], b[keep], c[keep], cr[keep], area[keep]
    nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
    i = rng.choice(len(area), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    fl = (u + v) > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    return a[i] + u * (b[i] - a[i]) + v * (c[i] - a[i]), nrm[i]

def basis(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    eye = np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)]) * DIST
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, np.stack([right, up, fwd], 1), fwd, right, up

def project(P, eye, M, res):
    Vc = (P - eye) @ M
    z = Vc[:, 2].clip(1e-6)
    f = 1.0 / math.tan(math.radians(FOV) / 2)
    x = (Vc[:, 0] * f / z * 0.5 + 0.5) * (res - 1)
    y = (-Vc[:, 1] * f / z * 0.5 + 0.5) * (res - 1)
    return x, y, z

def _zmin(flat, z, npix):
    """Per-pixel nearest depth. np.minimum.at is unbuffered and ~50x slower than
    sorting by depth and taking each pixel's first occurrence, which is what this
    does; the scoring loop runs it ~10k times so the difference is the whole job."""
    o = np.argsort(z, kind='stable')
    fs = flat[o]
    _, first = np.unique(fs, return_index=True)
    zb = np.full(npix, np.inf)
    zb[fs[first]] = z[o][first]
    return zb

def exposed(P, N, yaw, pitch, res=SCORE_RES):
    """fraction of surface area that is unoccluded AND front-facing"""
    eye, M, fwd, _, _ = basis(yaw, pitch)
    x, y, z = project(P, eye, M, res)
    face = (N @ fwd) < 0
    ok = face & (x >= 0) & (x < res) & (y >= 0) & (y < res)
    if not ok.any(): return 0.0
    xi, yi, zi = x[ok].astype(int), y[ok].astype(int), z[ok]
    flat = yi * res + xi
    zb = _zmin(flat, zi, res * res)
    return float((zi <= zb[flat] + 3e-3).sum()) / len(P)

def render(P, N, yaw, pitch, res=RES):
    eye, M, fwd, right, up = basis(yaw, pitch)
    L1 = -0.4 * right + 0.5 * up - 0.8 * fwd; L1 /= np.linalg.norm(L1)
    L2 = 0.7 * right + 0.2 * up - 0.5 * fwd;  L2 /= np.linalg.norm(L2)
    x, y, z = project(P, eye, M, res)
    face = (N @ fwd) < 0
    ok = face & (x >= 0) & (x < res) & (y >= 0) & (y < res)
    xi, yi, zi, ni = x[ok].astype(int), y[ok].astype(int), z[ok], N[ok]
    flat = yi * res + xi
    o = np.argsort(zi, kind='stable')                 # nearest sample per pixel wins
    fs = flat[o]
    _, first = np.unique(fs, return_index=True)
    idx = o[first]
    lam = 0.30 + 0.55 * np.clip(ni[idx] @ L1, 0, 1) + 0.25 * np.clip(ni[idx] @ L2, 0, 1)
    img = np.ones(res * res)
    img[fs[first]] = np.clip(0.62 * lam, 0, 1)
    hit = np.zeros(res * res, bool); hit[fs[first]] = True
    img, hit = img.reshape(res, res), hit.reshape(res, res)
    solid = binary_closing(hit, np.ones((3, 3)))      # seal splat pinholes
    img = np.where(solid & ~hit, 0.62, img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

# Up-axis read off orient_sheet_*.png by eye, one per mesh. Searching over up-axis
# as well as yaw maximises exposed area by laying animals on their backs, which is
# not a "front view" -- so up is FIXED here and only the yaw is optimised.
UP_FIX = {
    'aircraft': '+Y', 'beetle': '+Y', 'blub': '+Y', 'car': '-Z', 'cello': '+Y',
    'dino1': '+Y', 'dog': '+Z', 'duck': '+Y', 'elephant': '+Z', 'elk': '+Y',
    'gargoyle': '-Z', 'giraffe': '+Y', 'goat': '+Y', 'guitar': '+Y',
    'ivysaur': '+Y', 'kitten': '+Y', 'moai': '+Y', 'pegaso': '+Y', 'pig': '-X',
    # bob is Keenan Crane's genus-1 fish, same family and same orientation as blub.
    # Without an entry here the up-axis is searched too, and the search lays him on
    # his side (+Z, 36.8%) -- more exposed area, but not a front view, which is the
    # exact failure this table exists to prevent.
    'bob': '+Y',
    'robot': '+Y', 'shark': '+Y', 'teddy': '+Y',
}
YAWS = range(0, 360, 10)
PITCHES = (5, 10, 20, 30)          # near-horizontal, with headroom for
                                   # flat subjects (aircraft, car, shark)

def process(path):
    V, F = load_obj(path)
    V = normalize(V)
    best = None
    ups = {UP_FIX[path.stem]: UPS[UP_FIX[path.stem]]} if path.stem in UP_FIX else UPS
    for uname, U in ups.items():
        Pu, Nu = sample(V @ U.T, F, NSAMP)
        for yaw in YAWS:
            for pitch in PITCHES:
                sc = exposed(Pu, Nu, yaw, pitch)
                if best is None or sc > best[0]:
                    best = (sc, uname, yaw, pitch)
    sc, uname, yaw, pitch = best
    P, N = sample(V @ UPS[uname].T, F, NRENDER, seed=1)
    Image.fromarray(render(P, N, yaw, pitch)).save(OUT / f'{path.stem}.png')
    return dict(mesh=path.stem, up=uname, yaw=yaw, pitch=pitch,
                exposed=round(sc, 4), verts=len(V), tris=len(F))

if __name__ == '__main__':
    import multiprocessing as mp
    todo = sorted(OBJ.glob(sys.argv[1] if len(sys.argv) > 1 else '*.obj'))
    print(f'{len(todo)} meshes, {len(UPS)}x{len(list(YAWS))}x{len(PITCHES)} views each', flush=True)
    with mp.Pool(min(8, len(todo))) as pool:
        rows = []
        for r in pool.imap_unordered(process, todo):
            rows.append(r)
            print(f"{r['mesh']:16s} up={r['up']:2s} yaw={r['yaw']:3d} pitch={r['pitch']:2d} "
                  f"exposed={r['exposed']*100:5.1f}%  ({r['tris']} tris)", flush=True)
    rows.sort(key=lambda r: -r['exposed'])
    (HERE / 'front_views.json').write_text(json.dumps(rows, indent=2))
    with open(HERE / 'front_views.tsv', 'w') as f:
        f.write('mesh\tup\tyaw\tpitch\texposed\tverts\ttris\n')
        for r in rows:
            f.write('\t'.join(str(r[k]) for k in
                    ('mesh','up','yaw','pitch','exposed','verts','tris')) + '\n')
    print(f'\nwrote {len(rows)} renders to {OUT}')
