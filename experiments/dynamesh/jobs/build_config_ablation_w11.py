#!/usr/bin/env python3
"""build_config_ablation_w11.py -- 3DV supplement: temporal-component configuration.

All components ON, rung27 the sole backbone; only the temporal component's
configuration varies. Two axes crossing at the shipped setting:

    window   W=1 (no blend) / 3 / 5 / 7 / 11        flavour held at temporal-only
    flavour  temporal-only / spatial->temporal / joint   window held at W=3

The shipped setting is now W=11 (v2_G), not W=3, so 'ours' moves. The flavour
axis is still measured at W=3 because st_/v3_ were only ever trained at D; that
is stated in the output rather than papered over, since a reader could otherwise
read the flavour rows as W=11 comparisons.

Same rule as build_component_ladder.py: temporal from out/TEXEL/<obj>_<arm>.json,
PSNR/SSIM from final_eval.json of the run that json names, common object set only.
"""
import json, os, statistics as st
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
ALL = [l.split('\t')[0] for l in
       open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]

ROWS = [('No blend (reference)',        'r27',     'window'),
        ('Temporal only, W=3',          'r27mcfm', 'window'),
        ('Temporal only, W=5',          'w5',      'window'),
        ('Temporal only, W=7',          'w7',      'window'),
        ('Temporal only, W=11 (ours)',  'w11',     'window'),
        ('Spatial -> temporal, W=3',    'stD',     'flavour'),
        ('Joint spatio-temporal, W=3',  'v3D',     'flavour')]
KEYS = ['psnr', 'ssim', 'flicker', 'accel', 'drift']
LOWER = {'flicker', 'accel'}
OURS = 'w11'

_e = {}
def evalj(run):
    run = run.split('/')[-1]
    if run not in _e:
        p = f'{E}/runs/{run}/final_eval.json'
        _e[run] = json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def cell(o, arm):
    p = f'{E}/out/TEXEL/{o}_{arm}.json'
    if not os.path.exists(p): return None
    t = json.load(open(p)); e = evalj(t['run'])
    if not e: return None
    return dict(psnr=e['final']['psnr_mean'], ssim=e['final']['ssim_mean'],
                flicker=t['texel_flicker'], accel=t['texel_jerk'],
                drift=t['texel_drift'])

PO = {arm: {o: cell(o, arm) for o in ALL} for _, arm, _ in ROWS}
OBJ = [o for o in ALL if all(PO[arm][o] for _, arm, _ in ROWS)]
DROP = [o for o in ALL if o not in OBJ]
for o in DROP:
    print(f'  dropped {o}: missing {[l for l, a, _ in ROWS if not PO[a][o]]}')
print(f'  common object set: {len(OBJ)}/{len(ALL)}')

M = {arm: {k: st.mean(PO[arm][o][k] for o in OBJ) for k in KEYS}
     for _, arm, _ in ROWS}

def sign_p(w, n):
    k = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def wins(other, k):
    """objects where OURS beats `other` on metric k"""
    w = sum(1 for o in OBJ
            if (PO[OURS][o][k] < PO[other][o][k]) == (k in LOWER))
    return w, sign_p(w, len(OBJ))

FMT = {'psnr': '{:.2f}', 'ssim': '{:.4f}', 'flicker': '{:.5f}',
       'accel': '{:.5f}', 'drift': '{:.4f}'}
print(f"\n{'row':<30}{'axis':<9}" + ''.join(f'{k:>11}' for k in KEYS))
for lbl, arm, axis in ROWS:
    print(f'{lbl:<30}{axis:<9}' + ''.join(FMT[k].format(M[arm][k]) .rjust(11) for k in KEYS))

print(f'\nours (W=11) vs each other row, sign test over {len(OBJ)} objects')
sign = {}
for lbl, arm, _ in ROWS:
    if arm == OURS: continue
    sign[arm] = {k: wins(arm, k) for k in KEYS}
    s = '  '.join(f'{k} {sign[arm][k][0]:2d}/{len(OBJ)} p={sign[arm][k][1]:.2g}'
                  for k in ('flicker', 'accel', 'psnr'))
    print(f'  vs {lbl:<28} {s}')

json.dump({'n_objects': len(OBJ), 'objects': OBJ, 'dropped': DROP, 'ours': OURS,
           'rows': [{'label': l, 'arm': a, 'axis': ax, **{k: M[a][k] for k in KEYS}}
                    for l, a, ax in ROWS],
           'sign': {a: {k: {'wins': v[0], 'n': len(OBJ), 'p': v[1]}
                        for k, v in d.items()} for a, d in sign.items()}},
          open(f'{E}/out/config_ablation_w11.json', 'w'), indent=1)
print(f'\n-> {E}/out/config_ablation_w11.json')
