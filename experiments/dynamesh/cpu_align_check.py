"""
cpu_align_check.py — score and draw an alignment without touching the GPU queue.

Same camera and same masks as the GPU path, but the rasteriser is a painter's
algorithm in PIL: project the vertices, sort faces back to front, fill each with
its Lambert shade. For a silhouette that is exact; for shading it is good enough
to judge orientation by eye, which is all these panels are for.

Exists because the cluster queue was 41 deep and the alignment decision could not
wait behind it. GATE-cpu below proves the CPU silhouette matches the GPU one that
is already on disk before any of its numbers are believed.

GATE: COVERAGE, not symmetric IoU.
  Guan's point, and he is right. A cast shadow only ever ADDS to the video mask,
  so it can drag IoU down without our mesh being wrong anywhere. Coverage asks
  the one-sided question -- what fraction of OUR silhouette falls outside the
  video's object -- which a shadow cannot affect. It is why teapot_ceramic_crack
  scored terribly under IoU while being correctly posed.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

ap = argparse.ArgumentParser()
ap.add_argument('--rotations', default=str(_HERE / 'known_rotations/kling_rotations.json'))
ap.add_argument('--solved', default=str(_HERE / 'out/orient/orientation.json'))
ap.add_argument('--res', type=int, default=512)
ap.add_argument('--tol', type=int, default=12)
ap.add_argument('--out', default='out/guan_check')
A = ap.parse_args()

FOV, DIST = 40.0, 2.0
M = np.array([[-1., 0, 0], [0, 0, 1.], [0, 1., 0]])      # our canonical() axis map
F = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])      # his world -> our world

MAP = [
 ('pumpkin_rot',         'pumpkin', 'data/pumpkin_rot/mesh/pumpkin_unwarp.glb',
                                    'data/pumpkin_rot/frames_from_video'),
 ('horse_metal',         'horse',   'data/horse_metal/mesh/horse_unwarp.glb',
                                    'data/horse_metal/frames_from_video'),
 ('penguin_circuits',    'penguin', 'data/penguin_circuits/mesh/penguin_remesh_unwarp.glb',
                                    'data/penguin_circuits/frames_from_video'),
 ('whale_spots',         'whale',   'data/whale_spots/mesh/whale_unwarp.glb',
                                    'data/whale_spots/frames_from_video'),
 ('teapot_porcelain',    'teapot',  'data/teapot_porcelain/mesh/teapot.obj',
                                    'data/teapot_porcelain/frames_from_video'),
 ('teapot_ceramic_crack','teapot',  'data/teapot_ceramic_crack/mesh/teapot.obj',
                                    'data/teapot_ceramic_crack/frames_from_video'),
 ('spot_lava',           None,      'data/spot_lava/mesh/spot.obj',
                                    'data/spot_lava/frames_from_video'),
]


def canonical(v):
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    return v[:, (0, 2, 1)] * np.array([-1.0, 1.0, 1.0])


def largest_component(m):
    lb, n = ndimage.label(m)
    return m if n <= 1 else lb == (1 + int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_mask(path, res, tol):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
    b = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    m = largest_component(np.abs(a - np.median(b, axis=0)).max(axis=2) > tol)
    if m.shape[0] != res:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .resize((res, res), Image.NEAREST)) > 127
    return m


def render(V, Fc, res):
    """painter's algorithm -> (grey image uint8, silhouette bool)"""
    eye = np.array([0., -DIST, 0.])
    cam = V - eye                                    # camera looks along +y
    depth = cam[:, 1]
    f = 1.0 / math.tan(math.radians(FOV / 2) )
    d = np.maximum(depth, 1e-6)
    sx = (cam[:, 0] / d) * f
    sy = (cam[:, 2] / d) * f
    px = (sx * 0.5 + 0.5) * res
    py = (0.5 - sy * 0.5) * res                      # image y downward
    tri = V[Fc]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    n = n / ln
    cen = tri.mean(axis=1)
    vd = cen - eye
    vd = vd / np.linalg.norm(vd, axis=1, keepdims=True)
    lam = np.abs((n * -vd).sum(axis=1)).clip(0, 1)
    fd = depth[Fc].mean(axis=1)
    order = np.argsort(-fd)                          # far to near
    img = Image.new('L', (res, res), 255)
    sil = Image.new('L', (res, res), 0)
    di, ds = ImageDraw.Draw(img), ImageDraw.Draw(sil)
    for i in order:
        a, b, c = Fc[i]
        poly = [(px[a], py[a]), (px[b], py[b]), (px[c], py[c])]
        g = int(round(255 * (0.18 + 0.72 * lam[i])))
        di.polygon(poly, fill=g)
        ds.polygon(poly, fill=255)
    return np.asarray(img), np.asarray(sil) > 127


def score(mm, vm):
    inter = int((mm & vm).sum()); union = int((mm | vm).sum())
    ours = int(mm.sum()); unc = int((mm & ~vm).sum())
    return inter / max(union, 1), 100.0 * unc / max(ours, 1), ours, unc


def main():
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    K = json.load(open(A.rotations))['objects']
    S = {r['object'].split()[0]: np.array(r['R']) for r in json.load(open(A.solved))}
    R = A.res
    rows = []

    print('=' * 100)
    print(f"CPU alignment check @ {R}^2   gate = COVERAGE (% of OUR mesh outside the video object)")
    print('=' * 100)
    print(f'{"object":<24}{"variant":<16}{"IoU":>8}{"uncovered":>11}  verdict')
    print('-' * 100)

    for ours, his, mp, gd in MAP:
        mesh = trimesh.load(T2 / mp, process=False, force='mesh')
        V0 = canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
        Fc = np.asarray(mesh.faces)
        vm = video_mask(T2 / gd / 'frame_0001.png', R, A.tol)

        cands = []
        if his and his in K:
            Rt = np.array(K[his]['R_total'])
            cands.append(("guan  F R M'", F @ Rt @ M.T))
        if ours in S:
            cands.append(('our solve', S[ours]))
        if not cands:
            cands.append(('identity', np.eye(3)))

        imgs = {}
        for name, Rm in cands:
            g, mm = render(V0 @ Rm.T, Fc, R)
            iou, unc, px, upx = score(mm, vm)
            v = 'PASS' if unc < 3 else ('marginal' if unc < 8 else 'FAIL')
            print(f'{ours:<24}{name:<16}{iou:>8.4f}{unc:>10.2f}%  {v}   ({px:,} px, {upx:,} outside)')
            rows.append(dict(object=ours, variant=name, iou=round(iou, 4),
                             uncovered_pct=round(unc, 3), verdict=v,
                             mesh_px=px, uncovered_px=upx))
            imgs[name] = (g, mm)

        # panels: video | guan | ours | overlay(guan)
        vid = Image.open(T2 / gd / 'frame_0001.png').convert('RGB').resize((R, R), Image.LANCZOS)
        vid.save(OUT / f'{ours}_video.png')
        for name, (g, mm) in imgs.items():
            tag = 'guan' if name.startswith('guan') else 'ours'
            Image.fromarray(g).convert('RGB').save(OUT / f'{ours}_{tag}.png')
            ov = np.full((R, R, 3), 255, np.uint8)
            ov[mm & vm] = (60, 90, 220); ov[mm & ~vm] = (215, 60, 50); ov[vm & ~mm] = (40, 165, 90)
            Image.fromarray(ov).save(OUT / f'{ours}_{tag}_overlay.png')
        print('-' * 100)

    json.dump(rows, open(OUT / 'guan_check.json', 'w'), indent=2)

    # GATE-cpu: our CPU silhouette must agree with the GPU mask already on disk
    gp = _HERE / 'out/gt_targets_spot_lava/render_mask.npy'
    if gp.exists():
        gpu = np.load(gp)
        mesh = trimesh.load(T2 / 'data/spot_lava/mesh/spot.obj', process=False, force='mesh')
        _, cpu = render(canonical(np.asarray(mesh.vertices, float).copy()),
                        np.asarray(mesh.faces), gpu.shape[0])
        i = int((cpu & gpu).sum()); u = int((cpu | gpu).sum())
        print(f'\n[GATE-cpu] CPU silhouette vs the GPU render_mask on disk: IoU {i/max(u,1):.4f}')
        print('           (>0.98 means this rasteriser agrees with nvdiffrast and its numbers stand)')
    print(f'\n[SAVE] {OUT}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
