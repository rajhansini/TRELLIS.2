#!/usr/bin/env python3
"""fullrate_table.py -- Table B: the four methods that natively run 150 frames.

VALIDATION GATE. Every object's job also recomputed the 21-instant numbers from
the same loaded frames. Before emitting anything this diffs those against
baselines4d/results/video_metrics_all.json. A loader that silently drifted
cannot produce a table. Tolerance is 12% because the published eval resizes with
LANCZOS from the same source; anything larger is a real disagreement.

COMMON OBJECT SET. A row is only comparable to another if both cover the same
objects, so every row is restricted to objects where all methods are present.
"""
import json, glob, os, statistics as st, sys
from math import comb

E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
B='/net/projects/ranalab/rajhansini/baselines4d'
METHODS=['GT','frozen','ours','MeshNCA','L4GM']
U=['diagA','diagB','diagC']
TOL=0.12

raw={}
for p in sorted(glob.glob(f'{E}/out/FULLRATE_W11/*.json')):
    raw.update(json.load(open(p)))
if not raw: sys.exit('no FULLRATE jsons yet')
pub=json.load(open(f'{B}/results/video_metrics_all.json'))['per_cell']

# ---------- gate ----------
bad=[]
for o,r in raw.items():
    for key,c in r['sub'].items():
        view,m=key.split('|')
        # 'ours' is the arm we deliberately changed (W=3 -> W=11), so it cannot be
        # gated against the W=3 publication. Every other method still is.
        if m=='ours': continue
        ref=pub.get(f'{o}|{view}|{m}')
        if not ref or c is None: continue
        for k in ('flicker','accel','drift'):
            if ref.get(k) and abs(c[k]-ref[k])/ref[k] > TOL:
                bad.append(f'{o}|{view}|{m}.{k}: mine={c[k]:.5f} published={ref[k]:.5f}')
print(f'objects: {len(raw)}/24')
print(f'validation gate: {"PASS" if not bad else "FAIL"}  ({len(bad)} cells outside {TOL:.0%})')
for b in bad[:10]: print('   X '+b)
if bad: sys.exit('gate failed -- not emitting a table')

objs=sorted(o for o,r in raw.items()
            if all(f'{v}|{m}' in r['full'] for v in ['train']+U for m in METHODS if not (m=='GT' and v!='train')))
print(f'common object set: {len(objs)}/{len(raw)}')

def sup(o,m,k): return raw[o]['full'][f'train|{m}'][k]
def uns(o,m,k): return st.mean(raw[o]['full'][f'{c}|{m}'][k] for c in U)
def sign_p(w,n):
    kk=max(w,n-w); return min(1.0,2*sum(comb(n,i) for i in range(kk,n+1))/2**n)

rows={}
for m in METHODS:
    r={}
    for k in ('flicker','accel','drift'):
        r['s_'+k]=st.mean(sup(o,m,k) for o in objs)
        r['u_'+k]=r['s_'+k] if m=='GT' else st.mean(uns(o,m,k) for o in objs)
    if m!='GT':
        r['psnr']=st.mean(sup(o,m,'psnr') for o in objs)
        r['ssim']=st.mean(sup(o,m,'ssim') for o in objs)
    rows[m]=r

print(f'\n{"method":<10}{"sFlick":>9}{"sAccel":>9}{"sDrift":>8}{"uFlick":>9}{"uAccel":>9}{"uDrift":>8}{"PSNR":>7}{"SSIM":>8}')
for m in METHODS:
    r=rows[m]
    p=f"{r['psnr']:>7.2f}{r['ssim']:>8.4f}" if 'psnr' in r else f"{'--':>7}{'--':>8}"
    print(f"{m:<10}{r['s_flicker']:>9.5f}{r['s_accel']:>9.5f}{r['s_drift']:>8.4f}"
          f"{r['u_flicker']:>9.5f}{r['u_accel']:>9.5f}{r['u_drift']:>8.4f}{p}")

print('\nours vs each, per object (sign test over objects)')
for m in ['frozen','MeshNCA','L4GM']:
    out=[]
    for scope,fn in (('sup',sup),('uns',uns)):
        for k in ('flicker','accel'):
            w=sum(1 for o in objs if fn(o,'ours',k)<fn(o,m,k))
            out.append(f'{scope}/{k} {w:2d}/{len(objs)} p={sign_p(w,len(objs)):.2g}')
    print(f'  vs {m:<9} ' + '   '.join(out))

print('\nstride effect -- same renders, 150 frames vs the published 21 instants')
for m in ['ours','frozen','MeshNCA','L4GM']:
    f150=st.mean(uns(o,m,'flicker') for o in objs)
    f21=st.mean(st.mean(raw[o]['sub'][f'{c}|{m}']['flicker'] for c in U) for o in objs)
    print(f'  {m:<9} unseen flicker  150fr {f150:.5f}   21inst {f21:.5f}   x{f21/f150:.2f}')
fr150=st.mean(uns(o,'frozen','flicker') for o in objs); ou150=st.mean(uns(o,'ours','flicker') for o in objs)
fr21=st.mean(st.mean(raw[o]['sub'][f'{c}|frozen']['flicker'] for c in U) for o in objs)
ou21=st.mean(st.mean(raw[o]['sub'][f'{c}|ours']['flicker'] for c in U) for o in objs)
print(f'\n  ours vs frozen, unseen flicker:  150fr {100*(ou150-fr150)/fr150:+.1f}%   21inst {100*(ou21-fr21)/fr21:+.1f}%')

json.dump({'n_objects':len(objs),'objects':objs,'rows':rows},
          open(f'{E}/out/fullrate_table_w11.json','w'),indent=1)
print(f'\n-> {E}/out/fullrate_table_w11.json')
