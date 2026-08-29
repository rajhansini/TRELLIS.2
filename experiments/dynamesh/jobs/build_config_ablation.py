#!/usr/bin/env python3
"""build_config_ablation.py -- Table 2 (3DV supplement): all components ON,
only the temporal component's configuration varies. rung27 is the sole backbone.

Two axes crossing at v2_D (ours):
    window   W=1 (no blend) / W=3 (v2_D) / W=5 (v2_E)
    flavour  temporal-only (v2_D) / spatial->temporal (st_D) / joint (v3_D)

Same rule as build_component_ladder.py: temporal from out/TEXEL/<obj>_<arm>.json,
PSNR/SSIM from final_eval.json of the run that json names, common object set only.
"""
import json, os, statistics as st
from math import comb

E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
ALL=[l.split('\t')[0] for l in open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]
ROWS=[('No blend (reference)','r27'),('Temporal only, W=3 (ours)','r27mcfm'),
      ('Temporal only, W=5','w5'),('Spatial -> temporal, W=3','stD'),
      ('Joint spatio-temporal, W=3','v3D')]
KEYS=['psnr','ssim','flicker','accel','drift']; LOWER={'flicker','accel'}

_e={}
def evalj(run):
    run=run.split('/')[-1]
    if run not in _e:
        p=f'{E}/runs/{run}/final_eval.json'
        _e[run]=json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def cell(o,arm):
    p=f'{E}/out/TEXEL/{o}_{arm}.json'
    if not os.path.exists(p): return None
    t=json.load(open(p)); ev=evalj(t['run'])
    if not ev: return None
    return dict(psnr=ev['final']['psnr_mean'],ssim=ev['final']['ssim_mean'],
                flicker=t['texel_flicker'],accel=t['texel_jerk'],drift=t['texel_drift'])

PO={a:{o:cell(o,a) for o in ALL} for _,a in ROWS}
OBJ=[o for o in ALL if all(PO[a][o] for _,a in ROWS)]
DROP=[o for o in ALL if o not in OBJ]
for o in DROP:
    print(f'  dropped {o}: missing {[l for l,a in ROWS if not PO[a][o]]}')
print(f'  common object set: {len(OBJ)}/{len(ALL)}')
M={a:{k:st.mean(PO[a][o][k] for o in OBJ) for k in KEYS} for _,a in ROWS}

def sign_p(w,n):
    kk=max(w,n-w); return min(1.0,2*sum(comb(n,i) for i in range(kk,n+1))/2**n)

FMT={'psnr':'{:.2f}','ssim':'{:.4f}','flicker':'{:.5f}','accel':'{:.5f}','drift':'{:.4f}'}
print(f"\n{'configuration':<28}"+''.join(f'{k:>11}' for k in KEYS))
for lbl,a in ROWS:
    print(f'{lbl:<28}'+''.join(f'{FMT[k].format(M[a][k]):>11}' for k in KEYS))

best={k:(min if k in LOWER else max)((a for _,a in ROWS),key=lambda a:M[a][k]) for k in KEYS}
print('\nbest per column: '+'  '.join(f'{k}={dict(ROWS)and [l for l,a in ROWS if a==best[k]][0]}' for k in KEYS))

print('\nours (v2_D) vs each other configuration, per object:')
for lbl,a in ROWS:
    if a=='r27mcfm': continue
    out=[]
    for k in ('psnr','flicker','accel'):
        w=sum(1 for o in OBJ if (PO['r27mcfm'][o][k]<PO[a][o][k])==(k in LOWER))
        out.append(f'{k} {w:2d}/{len(OBJ)} p={sign_p(w,len(OBJ)):.2g}')
    d=100*(M['r27mcfm']['flicker']-M[a]['flicker'])/M[a]['flicker']
    print(f'  vs {lbl:<28} '+'  '.join(out)+f'   dFlicker {d:+.1f}%')

json.dump({'n_objects':len(OBJ),'objects':OBJ,'dropped':DROP,
           'rows':[{'label':l,'arm':a,**{k:M[a][k] for k in KEYS}} for l,a in ROWS]},
          open(f'{E}/out/config_ablation.json','w'),indent=1)
print(f'\n-> {E}/out/config_ablation.json')
