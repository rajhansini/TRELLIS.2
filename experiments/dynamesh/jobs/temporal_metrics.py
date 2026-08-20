"""temporal_metrics.py — does MCFM buy temporal coherence?

PSNR and SSIM are computed per frame against a per-frame target, so a method whose
only contribution is consistency BETWEEN frames is invisible to them. Measured:
MCFM moves PSNR by +0.021 dB mean (5/8 objects) and SSIM by +0.0005 (5/8) --
indistinguishable from the 1.55 dB run-to-run noise floor. That is not evidence
MCFM does nothing; it is evidence the wrong axis was measured.

THE METRIC. Standard temporal-consistency practice (Lai et al., ECCV 2018) warps
frame t-1 into frame t with optical flow and measures the residual. Here the warp
is the IDENTITY and that is exact, not an approximation: the camera is locked
(--turns 0) and the geometry is a fixed mesh, so nothing in the image moves. The
only thing that changes between consecutive frames is the texture. Removing the
flow step also removes flow-estimation error from the measurement.

  Flicker      F = mean_t mean_mask | P_t - P_{t-1} |
                   how much the render jitters frame to frame.

  Warp error   E = mean_t mean_mask | (P_t - P_{t-1}) - (G_t - G_{t-1}) |
                   how much the render's temporal change DEVIATES from the
                   target's. Only defined at yaw 0, the one view with a target.

  Evolution    D = mean_mask | P_last - P_first |
                   THE GUARD. Lower flicker is trivial to obtain by freezing the
                   texture, which would be a worse result reported as a better
                   number. A drop in F is only meaningful if D holds -- the
                   texture must still travel as far over the sequence.

Reported on the ADAPTED panel (right half of each frame; the left half is the
frozen arm, which also differs between the two runs because MCFM changes the
conditioning both arms are fed).
"""
import json, sys
import numpy as np
from pathlib import Path
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
O = E / 'out'
BAR, RES = 28, 518
OBJS = [('spot_lava', '', 150), ('teapot_lava2', '', 150),
        ('horse_metal', '_guan', 121), ('penguin_circuits', '_guan', 121),
        ('pumpkin_rot', '_guan', 121), ('whale_spots', '_guan', 121),
        ('teapot_porcelain', '_guan', 121), ('teapot_ceramic_crack', '_guan', 121)]
YAWS = [0, 90, 180, 270]


def adapted(p):
    """right half, label bar cropped, float [0,1]"""
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0
    return a[BAR:, a.shape[1] // 2:]


def mask_for(obj, suf):
    m = np.load(O / f'gt_targets_{obj}{suf}/render_mask.npy')
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                   .resize((RES, RES), Image.NEAREST)) > 127
    return m


def gt_frames(obj, suf, n):
    d = O / f'gt_targets_{obj}{suf}/frames'
    out = []
    for i in range(1, n + 1):
        g = Image.open(d / f'gt_{i:04d}.png').convert('RGB').resize((RES, RES), Image.LANCZOS)
        out.append(np.asarray(g, np.float32) / 255.0)
    return out


res = {}
for obj, suf, N in OBJS:
    m = mask_for(obj, suf)
    G = None
    for arm in ('r27', 'mcfm'):
        for y in YAWS:
            d = O / f'fixview_{obj}_{arm}_yaw{y}/frames'
            fr = sorted(d.glob('*.png'))
            if len(fr) < N:
                print(f'  skip {obj} {arm} yaw{y}: {len(fr)}/{N}'); continue
            P = [adapted(p) for p in fr[:N]]
            dP = [np.abs(P[t] - P[t - 1])[m].mean() for t in range(1, N)]
            F = float(np.mean(dP))
            D = float(np.abs(P[-1] - P[0])[m].mean())
            rec = {'flicker': F, 'evolution': D}
            if y == 0:
                if G is None:
                    G = gt_frames(obj, suf, N)
                dG = [(G[t] - G[t - 1]) for t in range(1, N)]
                Ew = float(np.mean([np.abs((P[t] - P[t - 1]) - dG[t - 1])[m].mean()
                                    for t in range(1, N)]))
                rec['warp_err'] = Ew
                rec['gt_flicker'] = float(np.mean([np.abs(x)[m].mean() for x in dG]))
            res.setdefault(obj, {}).setdefault(arm, {})[y] = rec
            print(f'  {obj:<22}{arm:<6}yaw{y:<4}F={F:.5f} D={D:.4f}'
                  + (f' Ewarp={rec["warp_err"]:.5f} (GT F={rec["gt_flicker"]:.5f})'
                     if y == 0 else ''), flush=True)

(O / 'temporal_metrics.json').write_text(json.dumps(res, indent=1))
print('\nwrote', O / 'temporal_metrics.json')
