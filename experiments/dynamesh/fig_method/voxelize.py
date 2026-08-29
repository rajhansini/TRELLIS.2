"""Voxelize spot.obj into a sparse SURFACE voxel set (the same thing TRELLIS's
SLat is: active voxels on/near the surface, ~2k of them), then emit isometric
cube polygons for the figure."""
import json, math, sys
import numpy as np

OBJ = '/net/projects/ranalab/rajhansini/TRELLIS.2/data/dynamesh_meshes/OBJ/spot.obj'

def load_obj(p):
    V, F = [], []
    for ln in open(p):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)

def sample_surface(V, F, n=600_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pr = area / area.sum()
    idx = rng.choice(len(F), size=n, p=pr)
    u = rng.random((n, 1)); v = rng.random((n, 1))
    flip = (u + v) > 1
    u[flip] = 1 - u[flip]; v[flip] = 1 - v[flip]
    return a[idx] + u * (b[idx] - a[idx]) + v * (c[idx] - a[idx])

def rot(deg, axis):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if axis == 'x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == 'y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def voxelize(P, n_long=20):
    lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    h = ext.max() / n_long                      # cubic voxels
    ijk = np.floor((P - lo) / h).astype(int)
    dims = ijk.max(0) + 1
    keep = np.unique(ijk, axis=0)
    return keep, dims

if __name__ == '__main__':
    V, F = load_obj(OBJ)
    P0 = sample_surface(V, F)
    for name, R in [('raw', np.eye(3)),
                    ('rx90', rot(90, 'x')),
                    ('rx-90', rot(-90, 'x')),
                    ('rz90', rot(90, 'z'))]:
        P = P0 @ R.T
        keep, dims = voxelize(P, 20)
        print(f'{name:6s} dims={tuple(dims)}  voxels={len(keep)}')


def voxelize_exact(V, F, n_long=13):
    """Conservative triangle-box voxelization: a cell is occupied iff some
    triangle actually overlaps it (13-axis separating-axis test).

    sample_surface()+voxelize() misses cells that a triangle crosses but no
    random sample lands in — 4 of 484 on spot.obj at n_long=13. Few, but they
    are holes in a figure whose entire point is that the voxel set never
    changes, so the figure uses this instead.
    """
    lo, hi = V.min(0), V.max(0)
    h = (hi - lo).max() / n_long
    dims = np.floor((V - lo) / h).astype(int).max(0) + 1
    occ = set()
    eps = 1e-12
    ax_unit = np.eye(3)
    for f in F:
        t = (V[f] - lo) / h                                  # triangle in cell units
        a = np.maximum(np.floor(t.min(0)).astype(int), 0)
        b = np.minimum(np.floor(t.max(0)).astype(int), dims - 1)
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                for k in range(a[2], b[2] + 1):
                    v = t - (np.array([i, j, k]) + 0.5)      # box centred at origin, half-extent 0.5
                    if (v.min(0) > 0.5 + eps).any() or (v.max(0) < -0.5 - eps).any():
                        continue
                    n = np.cross(v[1] - v[0], v[2] - v[0])   # triangle-plane axis
                    if abs(float(n @ v[0])) - 0.5 * np.abs(n).sum() > eps:
                        continue
                    for e in range(3):                       # 9 edge-cross axes
                        ed = v[(e + 1) % 3] - v[e]
                        for c in range(3):
                            p = np.cross(ax_unit[c], ed)
                            if np.abs(p).sum() < eps:
                                continue
                            d = v @ p
                            if d.min() > 0.5 * np.abs(p).sum() + eps or \
                               d.max() < -0.5 * np.abs(p).sum() - eps:
                                break
                        else:
                            continue
                        break
                    else:
                        occ.add((i, j, k))
    return np.array(sorted(occ), dtype=int), dims


def cull_hidden(vox):
    """Drop voxels no camera ray reaches. In this isometric view the +x, +y and
    +z neighbours cover a cube's right, top and left faces respectively, and all
    three sort in front of it, so a voxel with all three present is invisible.
    Removes ~40% of the polygons and lets every remaining cube be drawn OPAQUE,
    which is what makes the shell read as one solid structure instead of a
    see-through mesh with apparent holes."""
    occ = set(map(tuple, vox))
    keep = [v for v in vox
            if not ((v[0] + 1, v[1], v[2]) in occ
                    and (v[0], v[1] + 1, v[2]) in occ
                    and (v[0], v[1], v[2] + 1) in occ)]
    return np.array(keep, dtype=int)
