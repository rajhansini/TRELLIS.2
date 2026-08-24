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
