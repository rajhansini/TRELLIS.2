"""solve_similarity.py — solve rotation AND a 2D similarity, jointly.

WHY THIS EXISTS. solve_orientation.py searches rotation ONLY: centring and uniform
scale are fixed by canonical(), on the finding that they are exact for spot_lava.
That holds when the still handed to Kling was rendered with our camera. For batch G
it was not -- the cars came back 1.24x our render -- and then the rotation search is
optimising the wrong objective: it picks the rotation that best fits a silhouette of
the WRONG SIZE. car_effect1 scored 0.773 that way and 0.897 once a similarity was
fitted afterwards, and the two pigs scored 0.711 / 0.825.

Fitting the similarity AFTER the rotation is settled cannot recover what the rotation
search lost. This solves them together: at every rotation the scale and image-plane
offset are re-fitted before the rotation is scored, so a rotation is judged on what it
could achieve at its own best size.

GEOMETRY. EXTRINSICS puts the camera at world +y looking back, with
    image x  <-  world x        image y  <-  -world z        depth  <-  world y
so an image-plane offset is a shift in world (x, z), and scale is about the centroid.
Both are applied to the canonical vertices, so the render path is untouched.

INITIALISATION IS ANALYTIC, not a grid: scale from the square root of the mask-area
ratio, offset from the centroid difference. That is one rasterisation per rotation
instead of a hundred, which is what makes the joint search affordable at all.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.argv_backup = list(sys.argv)
sys.argv = [sys.argv[0]]                     # solve_orientation parses at import
import solve_orientation as SO               # noqa: E402
sys.argv = sys.argv_backup

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--mesh', required=True)
ap.add_argument('--frames', required=True)
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--search-res', type=int, default=384)
ap.add_argument('--frame', type=int, default=1)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--out', default='out/orient_similarity')
A = ap.parse_args()

T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
DEV = 'cuda'
# world units spanned by the full image at the mesh's depth (camera at z=2, fov 40)
WORLD_PER_IMG = 2.0 * math.tan(math.radians(20.0)) * 2.0


def fit_similarity(R, V, tv, res, iters=3):
    """Return (iou, s, tx, tz) with scale and image offset fitted for THIS rotation."""
    s, tx, tz = 1.0, 0.0, 0.0
    tv_np = tv.cpu().numpy()
    av = tv_np.sum()
    ys, xs = np.nonzero(tv_np)
    cv = (xs.mean() / res, ys.mean() / res) if av else (0.5, 0.5)
    best = (-1.0, s, tx, tz)
    for _ in range(iters):
        m = R.mask(V * s + np.array([tx, 0.0, tz]), res)
        mn = m.cpu().numpy()
        am = mn.sum()
        if am == 0:
            break
        iou = SO.iou_t(m, tv)
        if iou > best[0]:
            best = (iou, s, tx, tz)
        ym, xm = np.nonzero(mn)
        cm = (xm.mean() / res, ym.mean() / res)
        s *= math.sqrt(av / am)                       # area ratio -> linear scale
        tx += (cv[0] - cm[0]) * WORLD_PER_IMG         # image +x  ->  world +x
        tz -= (cv[1] - cm[1]) * WORLD_PER_IMG         # image +y  ->  world -z
    m = R.mask(V * s + np.array([tx, 0.0, tz]), res)
    iou = SO.iou_t(m, tv)
    return max(best, (iou, s, tx, tz))


def main():
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(T2 / A.mesh if not A.mesh.startswith('/') else A.mesh,
                        process=False, force='mesh')
    V0 = SO.canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
    R = SO.Raster(np.asarray(mesh.faces))
    fp = (T2 / A.frames / f'frame_{A.frame:04d}.png')
    vm_s, _ = SO.video_mask(fp, A.search_res, A.tol)
    vm_hi, _ = SO.video_mask(fp, A.res, A.tol)
    tv_s = torch.from_numpy(vm_s).to(DEV); tv_hi = torch.from_numpy(vm_hi).to(DEV)

    CUBE = SO.cube_rotations()
    print(f'== {A.name}: 24 cube rotations, similarity re-fitted at each', flush=True)
    coarse = []
    for i, M in enumerate(CUBE):
        iou, s, tx, tz = fit_similarity(R, V0 @ M.T, tv_s, A.search_res)
        coarse.append((iou, i, s, tx, tz))
    coarse.sort(reverse=True)
    print(f'   coarse best {coarse[0][0]:.4f} (scale {coarse[0][2]:.3f})  '
          f'runner-up {coarse[1][0]:.4f}', flush=True)

    best = (-1.0, None, 1.0, 0.0, 0.0)
    for iou0, i, *_ in coarse[:3]:
        M = CUBE[i].copy(); cur = iou0
        for step, rng in ((4.0, 16.0), (1.0, 4.0)):
            improved = True
            while improved:
                improved = False
                for axis in range(3):
                    for d in np.arange(-rng, rng + 1e-9, step):
                        if d == 0:
                            continue
                        e = [0.0, 0.0, 0.0]; e[axis] = float(d)
                        cand = SO.euler(*e) @ M
                        sc, s, tx, tz = fit_similarity(R, V0 @ cand.T, tv_s, A.search_res)
                        if sc > cur + 1e-6:
                            cur, M, improved = sc, cand, True
        sc, s, tx, tz = fit_similarity(R, V0 @ M.T, tv_s, A.search_res)
        if sc > best[0]:
            best = (sc, M, s, tx, tz)

    _, M, s, tx, tz = best
    iou_hi, s_hi, tx_hi, tz_hi = fit_similarity(R, V0 @ M.T, tv_hi, A.res, iters=4)
    rot_only = SO.iou_t(R.mask(V0 @ M.T, A.res), tv_hi)
    inter = R.mask(V0 @ M.T * s_hi + np.array([tx_hi, 0.0, tz_hi]), A.res)
    uncov = 1.0 - (inter & tv_hi).sum().item() / max(tv_hi.sum().item(), 1)
    print(f'   FULL-RES  IoU {iou_hi:.4f}   scale {s_hi:.4f}  '
          f'offset ({tx_hi:+.4f}, {tz_hi:+.4f})   uncovered {100*uncov:.2f}%')
    print(f'   rotation alone at that pose: {rot_only:.4f}   '
          f'(solve_orientation.py reported the rotation-only number)')
    rec = dict(name=A.name, mesh=A.mesh, iou=iou_hi, rotation_only_iou=rot_only,
               scale=s_hi, tx=tx_hi, tz=tz_hi, uncovered=uncov, R=M.tolist())
    (OUT / f'{A.name}.json').write_text(json.dumps(rec, indent=1))
    a = np.stack([vm_hi, inter.cpu().numpy(), np.zeros_like(vm_hi)], -1).astype(np.uint8) * 255
    Image.fromarray(a).resize((480, 480), Image.NEAREST).save(
        OUT / f'{A.name}_{iou_hi:.3f}.png')
    print(f'   wrote {OUT}/{A.name}.json', flush=True)


main()
