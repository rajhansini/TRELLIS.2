"""compare_bench_arms.py — are the sequential and frame-batched arms the same?

Batching changes the reduction order inside the GPU kernels, so bit-identical
output is NOT expected and its absence is not a bug. What must hold is that the
two arms are visually indistinguishable. Reported per view:

    PSNR(A,B)        > 50 dB  is indistinguishable
    max |A-B|        in 0-255 levels
    mean |A-B|
    % pixels differing by more than 1 level

usage: compare_bench_arms.py OUT_A OUT_B [--json out.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('a'); ap.add_argument('b')
ap.add_argument('--json', default=None)
A = ap.parse_args()

fa, fb = Path(A.a) / 'frames', Path(A.b) / 'frames'
views = sorted(p.name for p in fa.iterdir() if p.is_dir())
if not views:
    sys.exit(f'no view dirs under {fa}')

out, worst = {}, 1e9
for v in views:
    pa = sorted((fa / v).glob('*.png'))
    pb = sorted((fb / v).glob('*.png'))
    if len(pa) != len(pb) or not pa:
        sys.exit(f'{v}: frame count mismatch A={len(pa)} B={len(pb)}')
    se = 0.0; n = 0; mx = 0; ad = 0.0; ndiff = 0
    for x, y in zip(pa, pb):
        if x.name != y.name:
            sys.exit(f'{v}: name mismatch {x.name} vs {y.name}')
        u = np.asarray(Image.open(x).convert('RGB')).astype(np.int16)
        w = np.asarray(Image.open(y).convert('RGB')).astype(np.int16)
        if u.shape != w.shape:
            sys.exit(f'{v}/{x.name}: shape {u.shape} vs {w.shape}')
        d = np.abs(u - w)
        se += float((d.astype(np.float64) ** 2).sum()); n += d.size
        mx = max(mx, int(d.max())); ad += float(d.sum()); ndiff += int((d > 1).sum())
    mse = se / n
    psnr = float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    out[v] = dict(frames=len(pa), psnr_db=(None if mse == 0 else round(psnr, 2)),
                  bit_identical=(mse == 0), max_abs_level=mx,
                  mean_abs_level=round(ad / n, 4),
                  pct_pixels_gt1_level=round(100.0 * ndiff / n, 4))
    worst = min(worst, psnr)
    print(f'{v}: frames={len(pa)}  PSNR={"inf (bit-identical)" if mse==0 else f"{psnr:.2f} dB"}'
          f'  max={mx}  mean={ad/n:.4f}  >1level={100.0*ndiff/n:.4f}%')

verdict = 'IDENTICAL' if worst == float('inf') else (
    'INDISTINGUISHABLE' if worst > 50 else 'DIFFERENT — investigate')
print(f'\nworst-view PSNR: {"inf" if worst==float("inf") else f"{worst:.2f} dB"}  ->  {verdict}')
if A.json:
    json.dump(dict(views=out, worst_psnr_db=(None if worst == float('inf') else round(worst, 2)),
                   verdict=verdict), open(A.json, 'w'), indent=2)
    print('wrote', A.json)
