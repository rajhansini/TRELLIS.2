"""
measure_flicker.py — does MCFM actually buy temporal stability?

PSNR CANNOT ANSWER THIS. It scores each frame against its own target
independently, so a sequence that is correct frame-by-frame but jitters between
frames scores identically to one that evolves smoothly. MCFM blends the
conditioning tokens ACROSS TIME, so temporal stability is the only axis on which
it could possibly show a gain. Measured on 8 objects, its PSNR effect is
-0.04 to +0.22 dB, mean +0.07 -- i.e. nothing. This asks the other question.

THE METRIC, AND WHY IT IS THE SECOND DIFFERENCE

  A first-order difference |x_t - x_{t-1}| is the WRONG measure here: the texture
  is SUPPOSED to change over time, so a large first difference may be exactly the
  intended lava spreading. It conflates signal with defect.

  The SECOND difference |x_{t+1} - 2*x_t + x_{t-1}| is the discrete acceleration.
  A texture evolving smoothly -- however fast -- has a small second difference,
  because smooth motion is locally linear in time. Flicker, popping and
  frame-to-frame instability are precisely high acceleration. So this isolates
  the defect from the intended change, which is what a flicker metric has to do.

  Both are reported: D1 tells you how much the texture moves, D2 how raggedly.

WHERE IT IS MEASURED

  Inside the object silhouette only -- the background is constant white and would
  dilute every number toward zero.

  At FOUR fixed azimuths. The camera never rotates within a sequence, so all
  temporal change is texture, not viewpoint. Crucially yaw 90/180/270 were never
  supervised: if temporal blending helps anywhere it should help most where the
  loss gave no guidance, so those columns are the real test.
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--objects', default='spot_lava:150,pumpkin_rot:121')
ap.add_argument('--yaws', default='0,90,180,270')
ap.add_argument('--arms', default='r27,mcfm')
ap.add_argument('--out', default='out/flicker.json')
A = ap.parse_args()


def load_seq(obj, arm, yaw, n):
    """adapted panel (right half), object pixels only, as float [0,1]"""
    d = _HERE / 'out' / f'fixview_{obj}_{arm}_yaw{yaw}' / 'frames'
    frames, mask = [], None
    for i in range(1, n + 1):
        im = Image.open(d / f'{i:04d}.png').convert('RGB')
        w, h = im.size
        a = np.asarray(im.crop((w // 2, 28, w, h)), np.float32) / 255.0
        frames.append(a)
        m = a.min(axis=2) < 0.93
        mask = m if mask is None else (mask | m)
    return np.stack(frames), mask


def stats(seq, mask):
    x = seq[:, mask, :]                              # [T, P, 3]
    d1 = np.abs(np.diff(x, axis=0)).mean()
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2]).mean()
    return float(255 * d1), float(255 * d2)


def main():
    objs = [(o.split(':')[0], int(o.split(':')[1])) for o in A.objects.split(',')]
    yaws = [int(y) for y in A.yaws.split(',')]
    arms = A.arms.split(',')
    rows = []

    print('=' * 92)
    print('TEMPORAL STABILITY — does MCFM reduce frame-to-frame jitter?')
    print('  D1 = mean |x_t - x_(t-1)|        how much the texture moves  (not a defect)')
    print('  D2 = mean |x_(t+1) -2x_t +x_(t-1)|  acceleration = FLICKER    (lower is better)')
    print('=' * 92)

    for obj, n in objs:
        print(f'\n{obj}  ({n} frames)')
        print(f"  {'view':<18}{'D1 r27':>9}{'D1 mcfm':>10}{'D2 r27':>10}{'D2 mcfm':>10}"
              f"{'flicker Δ':>12}   verdict")
        print('  ' + '-' * 86)
        for y in yaws:
            s = {}
            for arm in arms:
                seq, mask = load_seq(obj, arm, y, n)
                s[arm] = stats(seq, mask)
            d1a, d2a = s['r27']; d1b, d2b = s['mcfm']
            rel = 100.0 * (d2b - d2a) / max(d2a, 1e-9)
            tag = 'supervised' if y == 0 else 'UNSEEN'
            verdict = ('MCFM smoother' if rel < -2 else
                       'MCFM rougher' if rel > 2 else 'no difference')
            print(f"  {str(y)+'° '+tag:<18}{d1a:>9.3f}{d1b:>10.3f}{d2a:>10.3f}{d2b:>10.3f}"
                  f"{rel:>+11.1f}%   {verdict}")
            rows.append(dict(object=obj, yaw=y, supervised=(y == 0),
                             d1_r27=round(d1a, 4), d1_mcfm=round(d1b, 4),
                             d2_r27=round(d2a, 4), d2_mcfm=round(d2b, 4),
                             flicker_rel_pct=round(rel, 2), verdict=verdict))

    json.dump(rows, open(_HERE / A.out, 'w'), indent=2)
    unseen = [r['flicker_rel_pct'] for r in rows if not r['supervised']]
    seen = [r['flicker_rel_pct'] for r in rows if r['supervised']]
    print('\n' + '=' * 92)
    print(f'mean flicker change, supervised view : {np.mean(seen):+.1f}%')
    print(f'mean flicker change, UNSEEN views    : {np.mean(unseen):+.1f}%   '
          f'({sum(1 for v in unseen if v < -2)}/{len(unseen)} smoother with MCFM)')
    print('negative = MCFM reduces flicker. This is the axis PSNR cannot see.')
    print(f'\n[SAVE] {A.out}\n[DONE]')


if __name__ == '__main__':
    main()
