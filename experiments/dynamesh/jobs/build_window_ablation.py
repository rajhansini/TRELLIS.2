#!/usr/bin/env python3
"""build_window_ablation.py -- the WINDOW axis at full width: W = 1 / 3 / 5 / 7 / 11.

The supplement's Table 2 crosses two axes at v2_D and only reaches W=5. This extends the
window axis alone, flavour held at temporal-only, so the curve can be read to its end
rather than stopped one point past our setting.

Same aggregation rule as build_config_ablation.py and build_component_ladder.py:
    temporal   from out/TEXEL/<obj>_<arm>.json
    PSNR/SSIM  from final_eval.json of the run THAT json names -- never re-resolved,
               so a cell's reconstruction and temporal numbers always come from one run
    objects    restricted to the common set where every row exists

texel_jerk is the JSON key for the SECOND difference, i.e. acceleration. The key name is
wrong and is kept only for compatibility; nothing here or in the paper calls it jerk.
"""
import json, os, statistics as st
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
# Object set. Defaults to the 24 in out/texel_r19_params.tsv, which is what the
# window axis covered before batches D and E were trained. OBJLIST=<tsv> overrides it
# with any file whose first column is the object name; jobs/rung37_objects.tsv carries
# all 42. The common-set restriction below still drops any object missing a row, so a
# wider list can never silently produce a ragged table.
_src = os.environ.get('OBJLIST', f'{E}/out/texel_r19_params.tsv')
_hdr = {'object', 'batch'}
ALL = [l.split('\t')[0] if 'rung37' not in _src else l.split('\t')[1]
       for l in open(_src).read().strip().split('\n')
       if l.split('\t')[0] not in _hdr]
ROWS = [('W=1   no blend (reference)', 'r27'),
        ('W=3   v2_D  (ours)',          'r27mcfm'),
        ('W=5   v2_E',                  'w5'),
        ('W=7   v2_F',                  'w7'),
        ('W=11  v2_G',                  'w11')]
KEYS = ['psnr', 'ssim', 'flicker', 'accel', 'drift']
LOWER = {'flicker', 'accel'}

_e = {}
def evalj(run):
    run = run.split('/')[-1]
    if run not in _e:
        p = f'{E}/runs/{run}/final_eval.json'
        _e[run] = json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def cell(o, arm):
    p = f'{E}/out/TEXEL/{o}_{arm}.json'
    if not os.path.exists(p):
        return None
    t = json.load(open(p)); ev = evalj(t['run'])
    if not ev:
        return None
    return dict(psnr=ev['final']['psnr_mean'], ssim=ev['final']['ssim_mean'],
                flicker=t['texel_flicker'], accel=t['texel_jerk'], drift=t['texel_drift'])

PO = {a: {o: cell(o, a) for o in ALL} for _, a in ROWS}
OBJ = [o for o in ALL if all(PO[a][o] for _, a in ROWS)]
DROP = [o for o in ALL if o not in OBJ]
for o in DROP:
    print(f'  dropped {o}: missing {[l.split()[0] for l, a in ROWS if not PO[a][o]]}')
print(f'  common object set: {len(OBJ)}/{len(ALL)}')
if not OBJ:
    raise SystemExit('no object has every window -- nothing to table yet')
M = {a: {k: st.mean(PO[a][o][k] for o in OBJ) for k in KEYS} for _, a in ROWS}

def sign_p(w, n):
    kk = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n)

FMT = {'psnr': '{:.2f}', 'ssim': '{:.4f}', 'flicker': '{:.5f}',
       'accel': '{:.5f}', 'drift': '{:.4f}'}
print(f"\n{'window':<28}" + ''.join(f'{k:>11}' for k in KEYS))
for lbl, a in ROWS:
    print(f'{lbl:<28}' + ''.join(f'{FMT[k].format(M[a][k]):>11}' for k in KEYS))

best = {k: (min if k in LOWER else max)((a for _, a in ROWS), key=lambda a: M[a][k])
        for k in KEYS}
lab = {a: l for l, a in ROWS}
print('\nbest per column (drift excluded -- it is a guard, never a score):')
for k in KEYS:
    if k == 'drift':
        continue
    print(f'  {k:<8} {lab[best[k]]}')

print('\nours (W=3) vs each other window, per object:')
for lbl, a in ROWS:
    if a == 'r27mcfm':
        continue
    out = []
    for k in ('psnr', 'flicker', 'accel'):
        w = sum(1 for o in OBJ if (PO['r27mcfm'][o][k] < PO[a][o][k]) == (k in LOWER))
        out.append(f'{k} {w:2d}/{len(OBJ)} p={sign_p(w, len(OBJ)):.2g}')
    d = 100 * (M['r27mcfm']['flicker'] - M[a]['flicker']) / M[a]['flicker']
    dd = 100 * (M['r27mcfm']['drift'] - M[a]['drift']) / M[a]['drift']
    print(f'  vs {lbl:<26} ' + '  '.join(out) + f'   dFlicker {d:+.1f}%  dDrift {dd:+.1f}%')

print('\nDRIFT GUARD. A window that lowers flicker AND lowers drift is damping the '
      'texture, not improving it. Read the dDrift column before calling any width a win.')

json.dump({'n_objects': len(OBJ), 'objects': OBJ, 'dropped': DROP,
           'rows': [{'label': l, 'arm': a, **{k: M[a][k] for k in KEYS}} for l, a in ROWS]},
          open(f'{E}/out/window_ablation.json', 'w'), indent=1)
print(f'\n-> {E}/out/window_ablation.json')
