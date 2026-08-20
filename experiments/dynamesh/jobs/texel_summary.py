"""texel_summary.py — collate the texel Temporal Flickering results.

One row per object: rung27 vs rung27+MCFM, measured on the PBR voxel field
(base_color per voxel) with no renderer in the path.

  flicker  1/(T-1) sum_t mean_vox |C_t - C_{t-1}|   VBench Temporal Flickering,
                                                    on texels
  jerk     mean_vox |C_{t+1} - 2C_t + C_{t-1}|      second order: jitter
  drift    mean_vox |C_T - C_1|                     THE GUARD -- flicker is
                                                    trivially reduced by freezing
                                                    the texture, so a drop only
                                                    counts if drift holds.
"""
import json
import numpy as np
from pathlib import Path

O = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/TEXEL')
pairs = json.load(open(O.parent / 'matched_pairs.json'))
rows = []
for p in pairs:
    a, b = O / f"{p['obj']}_r27.json", O / f"{p['obj']}_mcfm.json"
    if not (a.exists() and b.exists()):
        continue
    A, B = json.load(open(a)), json.load(open(b))
    if A['n_voxels'] != B['n_voxels']:
        print(f"  {p['obj']}: voxel count differs ({A['n_voxels']} vs {B['n_voxels']}) — SKIPPED")
        continue
    rows.append((p['obj'], A, B))

if not rows:
    print('no completed pairs yet'); raise SystemExit
print(f"{'object':<26}{'flicker r27':>12}{'mcfm':>11}{'dF %':>8}"
      f"{'jerk r27':>11}{'mcfm':>11}{'dJ %':>8}{'drift d%':>10}")
dF, dJ, dD = [], [], []
for o, A, B in rows:
    f = 100 * (B['texel_flicker'] - A['texel_flicker']) / A['texel_flicker']
    j = 100 * (B['texel_jerk'] - A['texel_jerk']) / A['texel_jerk']
    d = 100 * (B['texel_drift'] - A['texel_drift']) / A['texel_drift']
    dF.append(f); dJ.append(j); dD.append(d)
    print(f"{o:<26}{A['texel_flicker']:>12.6f}{B['texel_flicker']:>11.6f}{f:>+8.1f}"
          f"{A['texel_jerk']:>11.6f}{B['texel_jerk']:>11.6f}{j:>+8.1f}{d:>+10.1f}")
n = len(rows)
print(f"\n  objects: {n}")
print(f"  TEXEL Temporal Flickering lower : {sum(1 for x in dF if x<0)}/{n}   mean {np.mean(dF):+.1f}%")
print(f"  TEXEL jerk lower                : {sum(1 for x in dJ if x<0)}/{n}   mean {np.mean(dJ):+.1f}%")
print(f"  TEXEL drift (guard)             : mean {np.mean(dD):+.1f}%   "
      f"({'HOLDS - not damping' if abs(np.mean(dD))<5 else 'MOVED - check for damping'})")
json.dump({o: {'r27': {k: v for k, v in A.items() if not k.endswith('per_frame')},
               'mcfm': {k: v for k, v in B.items() if not k.endswith('per_frame')}}
           for o, A, B in rows}, open(O.parent / 'texel_summary.json', 'w'), indent=1)
print(f"\nwrote {O.parent / 'texel_summary.json'}")
