#!/usr/bin/env python3
"""build_component_ladder_arm.py -- component ablation with a SELECTABLE temporal arm.

Identical to build_component_ladder.py in every respect (same rows, same frozen
source, same common-object rule, same metrics, same sign tests). The ONLY thing
parameterised is which arm supplies the '+ temporal' row, so the W=3 ladder and
the W=11 ladder are produced by one code path and are directly comparable:

    --temporal-arm r27mcfm   '+ temporal' = rung27 + --mcfm v2_D  (W=3, published)
    --temporal-arm w11       '+ temporal' = rung27 + --mcfm v2_G  (W=11)

FROZEN still does NOT come from out/TEXEL/*_FROZEN.json -- three of those files
are wrong. It comes from the frozen_* fields of the r19 pass, as in the original.

COMMON-OBJECT SET is recomputed per invocation over that invocation's four rows,
so a W=11 ladder is internally consistent even if its object coverage differs
from the W=3 one. The two ladders are only comparable row-to-row when they report
the same n; the printout states it.
"""
import argparse, json, os, statistics as st
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'

ap = argparse.ArgumentParser()
ap.add_argument('--temporal-arm', default='r27mcfm',
                help="arm for the '+ temporal' row: r27mcfm=W3(v2_D), "
                     "w5=W5(v2_E), w7=W7(v2_F), w11=W11(v2_G)")
ap.add_argument('--label', required=True, help='output -> out/component_ladder_<label>.json')
ap.add_argument('--objs', default='', help='optional comma-separated object subset')
A = ap.parse_args()

ALL = [l.split('\t')[0] for l in
       open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]
if A.objs:
    want = {o for o in A.objs.split(',') if o}
    ALL = [o for o in ALL if o in want]

ROWS = [('frozen',            None),
        ('+ cross-attention', 'r19'),
        ('+ self-attention',  'r27'),
        ('+ temporal',        A.temporal_arm)]
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
for o in DROP:
    why = [lbl for lbl, arm in ROWS if not PO[arm][o]]
    print(f'  dropped {o}: no measurement for {why}')
print(f'  temporal arm: {A.temporal_arm}')
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
       'temporal_arm': A.temporal_arm,
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
json.dump(out, open(f'{E}/out/component_ladder_{A.label}.json', 'w'), indent=1)

print(f"\n{'row':<20}" + ''.join(f'{k:>11}' for k in KEYS))
for lbl, arm in ROWS:
    print(f'{lbl:<20}' + ''.join(f'{FMT[k].format(M[arm][k]):>11}' for k in KEYS))
print('\nstep deltas (wins = objects improved / n, sign-test p):')
for s in out['steps']:
    print(f"  {s['from']} -> {s['to']}")
    for k in KEYS:
        d = s[k]
        print(f"      {k:<8} {d['delta_pct']:+7.1f}%   {d['wins']:2d}/{d['n']}   p={d['p']:.2g}")
print(f'\n-> {E}/out/component_ladder_{A.label}.json')
