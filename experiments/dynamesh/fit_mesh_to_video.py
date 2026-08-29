"""
fit_mesh_to_video.py — make our rendered silhouette fit INSIDE the video's object.

make_gt_targets.py copies video[p] -> target[p] at the same screen location. That
is only valid where the video actually has the object. Wherever our silhouette
sticks out past it, the pixel is filled from its nearest video neighbour, and if
that neighbour is far away the fill becomes visible radial streaking.

So the objective is CONTAINMENT, not IoU. We want ours to be a subset of video.
Overshooting slightly (ours a little too small) costs a few supervised pixels;
undershooting (ours too big) costs correctness.

Sweeps an isotropic scale s and a vertical offset dy applied to the NORMALISED
mesh, scores each against the worst of several frames, and bakes the winner into
a new .obj so the training loop can rasterise it directly as v_raw.
"""
import argparse, math
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--frames-dir', required=True)
ap.add_argument('--probe-frames', default='1,75,150')
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--bg-thresh', type=float, default=245 / 255)
ap.add_argument('--s-range', default='0.60,1.00,21')
ap.add_argument('--dy-range', default='-0.20,0.20,21')
ap.add_argument('--max-fill', type=float, default=0.02,
                help='accept the largest silhouette whose uncovered fraction is '
                     'below this at EVERY probe frame')
ap.add_argument('--raw', action='store_true',
                help='mesh is already in the render frame (orientation and\n'
                     'normalisation already baked in); skip normalise()')
ap.add_argument('--out', required=True)
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
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.
    return r


def normalise(v):
    vmin, vmax = v.min(0), v.max(0)
    v = (v - (vmin + vmax) / 2) * (0.99999 / (vmax - vmin).max())
    t = v[:, 1].copy(); v[:, 1] = -v[:, 2]; v[:, 2] = t
    return v


def video_mask(path, res, thr):
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255
    m = a.min(2) < thr
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    return np.array(Image.fromarray((m * 255).astype(np.uint8))
                    .resize((res, res), Image.NEAREST)) > 127


def main():
    import nvdiffrast.torch as dr
    # file_type is pinned because the pristine input is '<name>.obj.prefit' (see
    # fit_align.sbatch). trimesh dispatches on the extension and this version raises
    # NotImplementedError on 'prefit'; the file is plain OBJ, so say so rather than
    # renaming a convention other objects (the penguins) already depend on.
    _ft = 'obj' if str(A.mesh).endswith('.prefit') else None
    mesh = trimesh.load(A.mesh, process=False, force='mesh',
                        **({'file_type': _ft} if _ft else {}))
    V0 = np.asarray(mesh.vertices, np.float64).copy()
    if not A.raw:
        V0 = normalise(V0)
    else:
        print('[MESH] --raw: taking the mesh as-is, already in the render frame')
    F = np.asarray(mesh.faces)
    ctx = dr.RasterizeCudaContext()
    f_t = torch.from_numpy(F).int().to(DEVICE).contiguous()
    full = (i2p(INTRINSICS.to(DEVICE), NEAR, FAR) @ EXTRINSICS.to(DEVICE)).unsqueeze(0)

    def render(s, dy):
        v = V0.copy() * s
        v[:, 2] += dy                       # camera-up axis after the swap
        t = torch.from_numpy(v).float().to(DEVICE)
        vh = torch.cat([t, torch.ones_like(t[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, f_t, (A.res, A.res))
        return (rast[0, ..., 3] > 0).cpu().numpy()

    probes = [int(x) for x in A.probe_frames.split(',')]
    vids = {p: video_mask(Path(A.frames_dir) / f'frame_{p:04d}.png', A.res, A.bg_thresh)
            for p in probes}
    for p in probes:
        print(f'[VIDEO] frame {p:3d}  {int(vids[p].sum()):,} px')

    s0, s1, sn = A.s_range.split(','); d0, d1, dn = A.dy_range.split(',')
    S = np.linspace(float(s0), float(s1), int(sn))
    DY = np.linspace(float(d0), float(d1), int(dn))

    best = None
    print(f'\n[SWEEP] {len(S)}×{len(DY)} = {len(S)*len(DY)} renders  '
          f'(objective: largest silhouette with fill < {100*A.max_fill:.1f}% on every probe)')
    for s in S:
        for dy in DY:
            ours = render(s, dy)
            a = int(ours.sum())
            if a < 1000:
                continue
            worst_fill = max(float((ours & ~vids[p]).sum()) / a for p in probes)
            if worst_fill <= A.max_fill and (best is None or a > best[0]):
                best = (a, s, dy, worst_fill)

    if best is None:
        print('\n[FAIL] no (s, dy) achieves the containment target. '
              'Loosen --max-fill or widen the ranges.')
        # report the best achievable anyway
        rows = []
        for s in S:
            for dy in DY:
                ours = render(s, dy); a = int(ours.sum())
                if a < 1000: continue
                wf = max(float((ours & ~vids[p]).sum()) / a for p in probes)
                rows.append((wf, a, s, dy))
        rows.sort()
        print('  best achievable fill fractions:')
        for wf, a, s, dy in rows[:5]:
            print(f'    s={s:.3f} dy={dy:+.3f}  area {a:,}  fill {100*wf:.2f}%')
        return

    a, s, dy, wf = best
    print(f'\n[BEST] s={s:.4f}  dy={dy:+.4f}  area {a:,} px  worst fill {100*wf:.2f}%')
    for p in probes:
        ours = render(s, dy)
        inter = int((ours & vids[p]).sum())
        print(f'   frame {p:3d}: covered {100*inter/a:.2f}% of ours   '
              f'ours/video {a/int(vids[p].sum()):.3f}')

    v = V0.copy() * s
    v[:, 2] += dy
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=F, process=False).export(str(out))
    print(f'\n[SAVE] {out}')
    print('  This mesh is ALREADY in the render frame: pass it as --mesh and the')
    print('  training loop rasterises it directly as v_raw.')


if __name__ == '__main__':
    main()
