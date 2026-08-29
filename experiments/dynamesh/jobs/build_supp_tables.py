#!/usr/bin/env python3
"""build_supp_tables.py -- the two supplementary ablation tables, same 24 objects
(batches A+B+C), same metric rule as build_component_ladder.py.

  Supp Table 1  FORWARD    frozen -> +CA -> +CA+SA -> +CA+SA+temporal
  Supp Table 2  REVERSED   frozen -> +temporal -> +temporal+CA -> +temporal+CA+SA

Both ladders share their endpoints by construction: row 1 is the same frozen
generator and row 4 is the same rung27+MCFM arm, so a disagreement between them is
attribution, not measurement.

ROW 2 OF THE REVERSED LADDER IS NOT A TRAINED ARM. It is the frozen model decoded
under the blend, read from the frozen_* fields of the r19mcfm jsons and the 'frozen'
block of that run's final_eval.json -- render_rung27_orbit.py gives the frozen and
adapted arms the SAME blended conditioning, so it is measured as a by-product.

  psnr/ssim  pixels, masked, from runs/<run>/final_eval.json
  flicker    texels, mean |C_t - C_{t-1}|
  accel      texels, mean |C_{t+1} - 2C_t + C_{t-1}|   (json field name: *_jerk)
  drift      texels, mean |C_T - C_1|                  GUARD, not a score
"""
import json, os, sys, statistics as st
from math import comb

E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
ALL=[l.split('\t')[0] for l in open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]
KEYS=['psnr','ssim','flicker','accel','drift']; LOWER={'flicker','accel'}
_e={}
def evalj(run):
    run=run.split('/')[-1]
    if run not in _e:
        p=f'{E}/runs/{run}/final_eval.json'
        _e[run]=json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def adapted(o,tag):
    """the trained arm of a texel json"""
    p=f'{E}/out/TEXEL/{o}_{tag}.json'
    if not os.path.exists(p): return None
    t=json.load(open(p)); ev=evalj(t['run'])
    if not ev or 'final' not in ev: return None
    return dict(psnr=ev['final']['psnr_mean'],ssim=ev['final']['ssim_mean'],
                flicker=t['texel_flicker'],accel=t['texel_jerk'],drift=t['texel_drift'])

def frozen(o,tag):
    """the frozen arm measured inside the SAME decode -- with that json's blend"""
    p=f'{E}/out/TEXEL/{o}_{tag}.json'
    if not os.path.exists(p): return None
    t=json.load(open(p))
    if 'frozen_flicker' not in t: return None
    ev=evalj(t['run'])
    if not ev or 'frozen' not in ev: return None
    return dict(psnr=ev['frozen']['psnr_mean'],ssim=ev['frozen']['ssim_mean'],
                flicker=t['frozen_flicker'],accel=t['frozen_jerk'],drift=t['frozen_drift'])

FWD=[('Frozen TRELLIS.2',        lambda o: frozen(o,'r19')),
     ('+ LoRA CA',               lambda o: adapted(o,'r19')),
     ('+ LoRA CA + LoRA SA',     lambda o: adapted(o,'r27')),
     ('+ LoRA CA + SA + Temporal',lambda o: adapted(o,'r27mcfm'))]
REV=[('Frozen TRELLIS.2',        lambda o: frozen(o,'r19')),
     ('+ Temporal',              lambda o: frozen(o,'r19mcfm')),
     ('+ Temporal + LoRA CA',    lambda o: adapted(o,'r19mcfm')),
     ('+ Temporal + CA + SA',    lambda o: adapted(o,'r27mcfm'))]

def build(name,rows):
    PO={i:{o:f(o) for o in ALL} for i,(_,f) in enumerate(rows)}
    OBJ=[o for o in ALL if all(PO[i][o] for i in range(len(rows)))]
    miss={lbl:[o for o in ALL if not PO[i][o]] for i,(lbl,_) in enumerate(rows)}
    print(f'\n{"="*72}\n{name}   common objects {len(OBJ)}/{len(ALL)}')
    for lbl,m in miss.items():
        if m: print(f'   missing {lbl}: {len(m)} -> {" ".join(m[:6])}{" ..." if len(m)>6 else ""}')
    if not OBJ: print('   NOT COMPUTABLE YET'); return None
    M={i:{k:st.mean(PO[i][o][k] for o in OBJ) for k in KEYS} for i in range(len(rows))}
    F={'psnr':'{:.2f}','ssim':'{:.4f}','flicker':'{:.5f}','accel':'{:.5f}','drift':'{:.4f}'}
    print(f'\n{"configuration":<30}'+''.join(f'{k:>11}' for k in KEYS))
    for i,(lbl,_) in enumerate(rows):
        print(f'{lbl:<30}'+''.join(f'{F[k].format(M[i][k]):>11}' for k in KEYS))
    def sp(w,n):
        kk=max(w,n-w); return min(1.0,2*sum(comb(n,j) for j in range(kk,n+1))/2**n)
    print('\nstep deltas (wins = objects improved / n, sign test):')
    steps=[]
    for i in range(len(rows)-1):
        s={'from':rows[i][0],'to':rows[i+1][0]}
        for k in KEYS:
            w=sum(1 for o in OBJ if (PO[i+1][o][k]<PO[i][o][k])==(k in LOWER))
            s[k]=dict(pct=100*(M[i+1][k]-M[i][k])/M[i][k],abs=M[i+1][k]-M[i][k],wins=w,n=len(OBJ),p=sp(w,len(OBJ)))
        steps.append(s)
        print(f'  {s["from"]} -> {s["to"]}')
        for k in KEYS:
            d=s[k]; print(f'      {k:<8} {d["pct"]:+7.1f}%  {d["wins"]:2d}/{d["n"]}  p={d["p"]:.2g}')
    return {'n_objects':len(OBJ),'objects':OBJ,
            'rows':[{'label':rows[i][0],**{k:M[i][k] for k in KEYS}} for i in range(len(rows))],
            'steps':steps,
            'per_object':{o:{rows[i][0]:PO[i][o] for i in range(len(rows))} for o in OBJ}}

out={'supp_table_1_forward':build('SUPPLEMENTARY TABLE 1 -- forward ladder',FWD),
     'supp_table_2_reversed':build('SUPPLEMENTARY TABLE 2 -- reversed ladder',REV)}
json.dump(out,open(f'{E}/out/supp_tables.json','w'),indent=1)
print(f'\n-> {E}/out/supp_tables.json')
