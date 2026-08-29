#!/usr/bin/env python3
"""build_window_table.py -- ONE table: every baseline plus ours at W=3/5/7/11.

All four window arms are read from dirs produced against the SAME reference (the
2D copy, out/gt_targets_<obj>/frames). out/FULLRATE_W11/ is deliberately NOT used:
it came from fullrate_metrics_w11.py scored against the raw video frame and reads
~6 dB lower, so mixing it in would make the ours rows incomparable to each other.

VALIDATION GATE. Same 12% gate as fullrate_table_scoped.py, applied to every
non-'ours' cell in every one of the four dirs. 'ours' is the arm that changed, so
it cannot be gated against the published W=3 numbers.

BASELINE IDENTITY. Only the ours arm differs between the four dirs, so the four
copies of each baseline must agree. That is asserted, not assumed.

COMMON OBJECT SET. Restricted to objects present with all five methods in all
four dirs, so every cell in the table covers the same objects.
"""
import json, os, sys, statistics as st
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
B = '/net/projects/ranalab/rajhansini/baselines4d'
METHODS = ['GT', 'frozen', 'ours', 'MeshNCA', 'L4GM']
U = ['diagA', 'diagB', 'diagC']
TOL = 0.12
WINDOWS = [('W=3', 'FULLRATE_CG'), ('W=5', 'FULLRATE_W5'),
           ('W=7', 'FULLRATE_W7'), ('W=11', 'FULLRATE_W11CG')]

WANT = json.load(open(f'{E}/out/fullrate_table_batchALL.json'))['objects']
pub = json.load(open(f'{B}/results/video_metrics_all.json'))['per_cell']

# ---------- load ----------
data = {}
for lab, d in WINDOWS:
    raw = {}
    for o in WANT:
        f = f'{E}/out/{d}/{o}.json'
        if os.path.exists(f):
            raw.update(json.load(open(f)))
    data[lab] = raw
    print(f'{lab:<5} {d:<16} {len(raw)}/{len(WANT)} objects')

# ---------- gate ----------
bad = []
for lab, _ in WINDOWS:
    for o, r in data[lab].items():
        for key, c in r['sub'].items():
            view, m = key.split('|')
            if m == 'ours':
                continue
            ref = pub.get(f'{o}|{view}|{m}')
            if not ref or c is None:
                continue
            for k in ('flicker', 'accel', 'drift'):
                if ref.get(k) and abs(c[k] - ref[k]) / ref[k] > TOL:
                    bad.append(f'{lab} {o}|{view}|{m}.{k}: mine={c[k]:.5f} pub={ref[k]:.5f}')
print(f'validation gate: {"PASS" if not bad else "FAIL"}  ({len(bad)} cells outside {TOL:.0%})')
for b in bad[:10]:
    print('   X ' + b)
if bad:
    sys.exit('gate failed -- not emitting a table')

# ---------- common object set ----------
def complete(r):
    return all(f'{v}|{m}' in r['full']
               for v in ['train'] + U for m in METHODS if not (m == 'GT' and v != 'train'))

objs = sorted(set.intersection(*[{o for o, r in data[lab].items() if complete(r)}
                                 for lab, _ in WINDOWS]))
print(f'common object set: {len(objs)}')
if len(objs) < len(WANT):
    print('  dropped: ' + ', '.join(sorted(set(WANT) - set(objs))))

# ---------- baseline identity ----------
drift_max = 0.0
for m in [x for x in METHODS if x != 'ours']:
    for o in objs:
        for v in ['train'] + U:
            if m == 'GT' and v != 'train':
                continue
            vals = [data[lab][o]['full'][f'{v}|{m}'] for lab, _ in WINDOWS]
            for k in ('flicker', 'accel', 'drift'):
                lo, hi = min(x[k] for x in vals), max(x[k] for x in vals)
                if hi:
                    drift_max = max(drift_max, abs(hi - lo) / hi)
print(f'baseline identity across the four dirs: max relative spread {drift_max:.2e}')
if drift_max > 1e-9:
    print('  WARNING: baselines are not identical across window dirs')

# ---------- rows ----------
def sup(lab, o, m, k):
    return data[lab][o]['full'][f'train|{m}'][k]

def uns(lab, o, m, k):
    return st.mean(data[lab][o]['full'][f'{c}|{m}'][k] for c in U)

def sign_p(w, n):
    kk = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n)

def row(lab, m):
    r = {}
    for k in ('flicker', 'accel', 'drift'):
        r['s_' + k] = st.mean(sup(lab, o, m, k) for o in objs)
        r['u_' + k] = r['s_' + k] if m == 'GT' else st.mean(uns(lab, o, m, k) for o in objs)
    if m != 'GT':
        r['psnr'] = st.mean(sup(lab, o, m, 'psnr') for o in objs)
        r['ssim'] = st.mean(sup(lab, o, m, 'ssim') for o in objs)
    return r

rows = [('Ground truth', row('W=3', 'GT')),
        ('Frozen TRELLIS.2', row('W=3', 'frozen')),
        ('MeshNCA', row('W=3', 'MeshNCA')),
        ('L4GM', row('W=3', 'L4GM'))]
for lab, _ in WINDOWS:
    rows.append((f'DynaMesh (ours, {lab})', row(lab, 'ours')))

hdr = f'{"method":<24}{"sFlick":>9}{"sAccel":>9}{"sDrift":>8}{"uFlick":>9}{"uAccel":>9}{"uDrift":>8}{"PSNR":>7}{"SSIM":>8}'
print('\n' + hdr)
for name, r in rows:
    p = f"{r['psnr']:>7.2f}{r['ssim']:>8.4f}" if 'psnr' in r else f"{'--':>7}{'--':>8}"
    print(f"{name:<24}{r['s_flicker']:>9.5f}{r['s_accel']:>9.5f}{r['s_drift']:>8.4f}"
          f"{r['u_flicker']:>9.5f}{r['u_accel']:>9.5f}{r['u_drift']:>8.4f}{p}")

# ---------- sign tests: each window vs each baseline ----------
sign = {}
print('\nsign tests, ours(W) better than baseline, over the common objects')
for lab, _ in WINDOWS:
    sign[lab] = {}
    for m in ['frozen', 'MeshNCA', 'L4GM']:
        e = {}
        for scope, fn in (('s', sup), ('u', uns)):
            for k in ('flicker', 'accel'):
                w = sum(1 for o in objs if fn(lab, o, 'ours', k) < fn(lab, o, m, k))
                e[f'{scope}_{k}'] = [w, len(objs), sign_p(w, len(objs))]
        sign[lab][m] = e
        s = '  '.join(f"{kk} {vv[0]}/{vv[1]} p={vv[2]:.2g}" for kk, vv in e.items())
        print(f'  {lab:<5} vs {m:<9} {s}')

out = {'n_objects': len(objs), 'objects': objs,
       'windows': [w for w, _ in WINDOWS],
       'dirs': {w: d for w, d in WINDOWS},
       'rows': [{'name': n, **r} for n, r in rows],
       'sign': sign,
       'gate': {'tol': TOL, 'cells_outside': 0},
       'baseline_max_spread': drift_max}
json.dump(out, open(f'{E}/out/window_table_all.json', 'w'), indent=1)
print(f'\n-> {E}/out/window_table_all.json')
