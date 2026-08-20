"""jitter_spectrum.py — separate JITTER from EVOLUTION. tLP cannot.

tLP and warping error are FIRST-difference quantities: |I_t - I_{t-1}|. They
collapse two very different things into one number --

    evolution  the texture legitimately progressing, frame over frame
    jitter     high-frequency back-and-forth that is pure artefact

-- so a method that removes jitter and a method that merely slows everything down
look identical to them. That ambiguity is exactly what is in dispute here, so
neither metric can settle it.

TWO DECOMPOSITIONS THAT CAN.

1. SECOND TEMPORAL DIFFERENCE (jerk).  a_t = I_{t+1} - 2 I_t + I_{t-1}
   The first difference is velocity; the second is acceleration. A texture
   evolving smoothly in one direction has a large first difference and a SMALL
   second. One jittering in place has a small first difference and a LARGE
   second. Reported as deviation from the reference's own jerk, |a_pred - a_gt|,
   so slowing down cannot win: matching GT is the target, not minimising.

2. TEMPORAL FREQUENCY SPECTRUM.  FFT along t, per pixel.
   Evolution is low-frequency, jitter is high. Split the spectrum at the midpoint
   and report the power in each band RELATIVE TO GT's power in that same band.
   The claim "MCFM removes jitter and keeps evolution" makes a precise, falsifiable
   prediction here: high-band ratio moves toward 1.0 while low-band ratio does not
   fall. If instead both bands drop together, it is damping, and this says so.
"""
import json
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
SUB = 4          # spatial stride; jitter is a temporal quantity, full res not needed


def adapted(p):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0
    return a[BAR:, a.shape[1] // 2:][::SUB, ::SUB].mean(-1)


def gt(obj, suf, i):
    g = Image.open(O / f'gt_targets_{obj}{suf}/frames/gt_{i:04d}.png').convert('L')
    return np.asarray(g.resize((RES, RES), Image.LANCZOS), np.float32)[::SUB, ::SUB] / 255.0


res = {}
for obj, suf, N in OBJS:
    m = np.load(O / f'gt_targets_{obj}{suf}/render_mask.npy')
    m = (np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                    .resize((RES, RES), Image.NEAREST)) > 127)[::SUB, ::SUB]
    G = np.stack([gt(obj, suf, i) for i in range(1, N + 1)])          # [T,H,W]
    row = {}
    for arm in ('r27', 'mcfm'):
        fr = sorted((O / f'fixview_{obj}_{arm}_yaw0/frames').glob('*.png'))[:N]
        if len(fr) < N:
            print(f'  skip {obj} {arm}'); continue
        P = np.stack([adapted(p) for p in fr])
        # 1. jerk, as deviation from GT's jerk
        jp = P[2:] - 2 * P[1:-1] + P[:-2]
        jg = G[2:] - 2 * G[1:-1] + G[:-2]
        jerk_dev = float(np.abs(jp - jg)[:, m].mean())
        jerk_raw = float(np.abs(jp)[:, m].mean())
        # 2. temporal spectrum, per pixel, mean-removed
        def bands(X):
            F = np.abs(np.fft.rfft(X - X.mean(0, keepdims=True), axis=0))
            k = F.shape[0]
            lo = F[1:max(2, k // 4)].mean(0)      # slow evolution
            hi = F[k // 2:].mean(0)               # jitter
            return float(lo[m].mean()), float(hi[m].mean())
        lp, hp = bands(P)
        row[arm] = {'jerk_dev': jerk_dev, 'jerk_raw': jerk_raw, 'lo': lp, 'hi': hp}
    lg, hg = bands(G)
    row['gt'] = {'jerk_raw': float(np.abs(jg)[:, m].mean()), 'lo': lg, 'hi': hg}
    if 'r27' in row and 'mcfm' in row:
        a, b, g = row['r27'], row['mcfm'], row['gt']
        print(f"  {obj:<21} jerkdev {a['jerk_dev']:.5f}->{b['jerk_dev']:.5f} "
              f"({100*(b['jerk_dev']-a['jerk_dev'])/a['jerk_dev']:+6.1f}%)   "
              f"HI/gt {a['hi']/g['hi']:.2f}->{b['hi']/g['hi']:.2f}   "
              f"LO/gt {a['lo']/g['lo']:.2f}->{b['lo']/g['lo']:.2f}", flush=True)
    res[obj] = row

(O / 'jitter_spectrum.json').write_text(json.dumps(res, indent=1))
print('\nwrote', O / 'jitter_spectrum.json')
