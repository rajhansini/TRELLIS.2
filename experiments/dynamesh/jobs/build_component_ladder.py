#!/usr/bin/env python3
"""build_component_ladder.py -- the CLEAN component ablation (Itai's table 1).

ONE ARM PER ROW. build_results_table.py averages every arm on each rung:
row (c) is r27+r28+r29+r30, row (d) is ten arms. That answers "does this rung
help on average", not "does this component earn its place" -- rungs 28/29/30 are
rung27 PLUS KL regularisers and 32/34 are a different operator (wide context), so
averaging them in dilutes the component with a second, unrelated axis.

    frozen                 frozen_* fields of the r19 pass
    + cross-attention      rung19   targets to_q,to_kv,to_out
    + self-attention       rung27   targets to_q,to_kv,to_out,sa_qkv,sa_out
    + temporal (ours)      rung27 + --mcfm v2_D   (parameter-free, 3-frame)

Each row differs from the one above it by EXACTLY ONE component.

FROZEN DOES NOT COME FROM out/TEXEL/*_FROZEN.json -- three of those 24 files are
wrong (spot_lava probed at 8 frames; hand_rorschach and pumpkin_rot disagree with
every other measurement of the same quantity by 36-56%). The frozen_* fields
embedded in the *_r19.json files are complete for all 24, at each object's own
frame count. Same rule as build_results_table.py.

COMMON-OBJECT SET. A row is only comparable to another if both cover the same
objects, so every row is restricted to objects where ALL FOUR rows exist. Objects
dropped for that reason are named in the output rather than silently excluded.

  psnr/ssim  pixels, masked, from runs/<run>/final_eval.json ['final']
  flicker    texels, mean |C_t - C_{t-1}|          on pbr.base_color[0:3]
  accel      texels, mean |C_{t+1} - 2C_t + C_{t-1}|
  drift      texels, mean |C_T - C_1|              GUARD, not a score
"""
import json, os, statistics as st
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
ALL = [l.split('\t')[0] for l in
       open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]

ROWS = [('frozen',            None),
        ('+ cross-attention', 'r19'),
        ('+ self-attention',  'r27'),
        ('+ temporal',        'r27mcfm')]
KEYS = ['psnr', 'ssim', 'flicker', 'accel', 'drift']
LOWER_BETTER = {'flicker', 'accel'}

_e = {}
def evalj(run):
    run = run.split('/')[-1]
    if run not in _e:
        p = f'{E}/runs/{run}/final_eval.json'
        _e[run] = json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def cell(o, arm):
    """One object, one arm -> dict of the five metrics, or None if not measured."""
    if arm is None:                                   # frozen, from the r19 pass
        p = f'{E}/out/TEXEL/{o}_r19.json'
        if not os.path.exists(p): return None
        t = json.load(open(p)); e = evalj(t['run'])
        if not e or 'frozen' not in e: return None
        return dict(psnr=e['frozen']['psnr_mean'], ssim=e['frozen']['ssim_mean'],
                    flicker=t['frozen_flicker'], accel=t['frozen_jerk'],
                    drift=t['frozen_drift'])
    p = f'{E}/out/TEXEL/{o}_{arm}.json'
    if not os.path.exists(p): return None
    t = json.load(open(p)); e = evalj(t['run'])
    if not e: return None
    return dict(psnr=e['final']['psnr_mean'], ssim=e['final']['ssim_mean'],
                flicker=t['texel_flicker'], accel=t['texel_jerk'],
                drift=t['texel_drift'])

PO = {arm: {o: cell(o, arm) for o in ALL} for _, arm in ROWS}
OBJ = [o for o in ALL if all(PO[arm][o] for _, arm in ROWS)]
DROP = [o for o in ALL if o not in OBJ]
if DROP:
    for o in DROP:
        why = [lbl for lbl, arm in ROWS if not PO[arm][o]]
        print(f'  dropped {o}: no measurement for {why}')
print(f'  common object set: {len(OBJ)}/{len(ALL)}')

M = {arm: {k: st.mean(PO[arm][o][k] for o in OBJ) for k in KEYS} for _, arm in ROWS}

def sign_p(w, n):
    k = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def wins(hi, lo, k):
    w = sum(1 for o in OBJ if (PO[lo][o][k] < PO[hi][o][k]) == (k in LOWER_BETTER))
    return w, sign_p(w, len(OBJ))

FMT = {'psnr': '{:.2f}', 'ssim': '{:.4f}', 'flicker': '{:.5f}',
       'accel': '{:.5f}', 'drift': '{:.4f}'}

out = {'n_objects': len(OBJ), 'objects': OBJ, 'dropped': DROP,
       'rows': [{'label': lbl, 'arm': arm or 'FROZEN',
                 **{k: M[arm][k] for k in KEYS}} for lbl, arm in ROWS],
       'steps': []}
for (l0, a0), (l1, a1) in zip(ROWS, ROWS[1:]):
    s = {'from': l0, 'to': l1}
    for k in KEYS:
        w, p = wins(a0, a1, k)
        s[k] = {'delta_pct': 100 * (M[a1][k] - M[a0][k]) / M[a0][k],
                'delta_abs': M[a1][k] - M[a0][k], 'wins': w, 'n': len(OBJ), 'p': p}
    out['steps'].append(s)
json.dump(out, open(f'{E}/out/component_ladder.json', 'w'), indent=1)

print(f"\n{'row':<20}" + ''.join(f'{k:>11}' for k in KEYS))
for lbl, arm in ROWS:
    print(f'{lbl:<20}' + ''.join(f'{FMT[k].format(M[arm][k]):>11}' for k in KEYS))
print('\nstep deltas (wins = objects improved / n, sign-test p):')
for s in out['steps']:
    print(f"  {s['from']} -> {s['to']}")
    for k in KEYS:
        d = s[k]
        print(f"      {k:<8} {d['delta_pct']:+7.1f}%   {d['wins']:2d}/{d['n']}   p={d['p']:.2g}")
print(f"\n-> {E}/out/component_ladder.json")
