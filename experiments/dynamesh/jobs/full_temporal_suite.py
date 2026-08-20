"""full_temporal_suite.py — the three temporal metrics over every matched pair.

  Temporal Flickering  (VBench, Huang et al., CVPR 2024)
      F = 1/(T-1) sum_t mean_M | I_t - I_{t-1} |
      Valid only on static scenes; ours are static by construction (camera locked,
      fixed mesh), so the only thing that can change between frames is texture.

  Jerk deviation from GT
      mean_M | a_pred - a_gt |,  a_t = I_{t+1} - 2 I_t + I_{t-1}
      Second-order, and referenced to the TARGET's own jerk, so a method cannot
      win it by slowing down -- which plain flicker can be won by.

  Band split + jitter correlation
      FFT along t per pixel; low band = evolution, high band = jitter, each as a
      ratio to GT's power in the same band. Plus corr(a_pred, a_gt): if the
      high-frequency content is uncorrelated with the target's it is noise, and
      removing it is a gain rather than a loss of detail.

Reported for r27 vs r27+mcfm on every pair sharing one gt_dir, mesh and frame count.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
O = E / 'out'
BAR, RES, SUB = 28, 518, 4
pairs = json.load(open(O / 'matched_pairs.json'))


def adapted(p):
    a = np.asarray(Image.open(p).convert('L'), np.float32) / 255.0
    return a[BAR:, a.shape[1] // 2:][::SUB, ::SUB]


def find_targets(obj):
    for suf in ('', '_guan'):
        d = O / f'gt_targets_{obj}{suf}'
        if (d / 'render_mask.npy').exists():
            return d
    return None


def bands(X, m):
    F = np.abs(np.fft.rfft(X - X.mean(0, keepdims=True), axis=0))
    k = F.shape[0]
    return (float(F[1:max(2, k // 4)].mean(0)[m].mean()), float(F[k // 2:].mean(0)[m].mean()))


res = {}
for p in pairs:
    obj, N = p['obj'], p['n']
    td = find_targets(obj)
    if td is None:
        print(f'  {obj}: no targets, skip'); continue
    dirs = {a: O / f'fixview_{obj}_{a}_yaw0/frames' for a in ('r27', 'mcfm')}
    if any(len(list(d.glob('*.png'))) < N for d in dirs.values()):
        print(f'  {obj}: renders incomplete, skip'); continue
    m = np.load(td / 'render_mask.npy')
    m = (np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                    .resize((RES, RES), Image.NEAREST)) > 127)[::SUB, ::SUB]
    G = np.stack([np.asarray(Image.open(td / f'frames/gt_{i:04d}.png').convert('L')
                             .resize((RES, RES), Image.LANCZOS), np.float32)[::SUB, ::SUB] / 255.
                  for i in range(1, N + 1)])
    jg = G[2:] - 2 * G[1:-1] + G[:-2]
    lg, hg = bands(G, m)
    row = {'n': N}
    for arm in ('r27', 'mcfm'):
        P = np.stack([adapted(q) for q in sorted(dirs[arm].glob('*.png'))[:N]])
        jp = P[2:] - 2 * P[1:-1] + P[:-2]
        lo, hi = bands(P, m)
        row[arm] = {
            'flicker': float(np.abs(np.diff(P, axis=0))[:, m].mean()),
            'jerk_dev': float(np.abs(jp - jg)[:, m].mean()),
            'lo_ratio': lo / lg, 'hi_ratio': hi / hg,
            'jerk_corr': float(np.corrcoef(jp[:, m].ravel(), jg[:, m].ravel())[0, 1]),
        }
    a, b = row['r27'], row['mcfm']
    row['dF'] = 100 * (b['flicker'] - a['flicker']) / a['flicker']
    row['dJ'] = 100 * (b['jerk_dev'] - a['jerk_dev']) / a['jerk_dev']
    res[obj] = row
    print(f"  {obj:<26} F {row['dF']:+7.1f}%  jerk {row['dJ']:+7.1f}%   "
          f"LO {a['lo_ratio']:.2f}->{b['lo_ratio']:.2f}  HI {a['hi_ratio']:.2f}->{b['hi_ratio']:.2f}"
          f"  corr {a['jerk_corr']:.2f}", flush=True)

(O / 'full_temporal_suite.json').write_text(json.dumps(res, indent=1))
if res:
    dF = [v['dF'] for v in res.values()]; dJ = [v['dJ'] for v in res.values()]
    lo_a = np.mean([v['r27']['lo_ratio'] for v in res.values()])
    lo_b = np.mean([v['mcfm']['lo_ratio'] for v in res.values()])
    print(f"\n  objects: {len(res)}")
    print(f"  Temporal Flickering lower : {sum(1 for x in dF if x<0)}/{len(dF)}   mean {np.mean(dF):+.1f}%")
    print(f"  Jerk deviation lower      : {sum(1 for x in dJ if x<0)}/{len(dJ)}   mean {np.mean(dJ):+.1f}%")
    print(f"  Evolution guard (LO/GT)   : {lo_a:.3f} -> {lo_b:.3f}   (unchanged = not damping)")
