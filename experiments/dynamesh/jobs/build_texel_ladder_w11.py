#!/usr/bin/env python3
"""Texel component ladder with the temporal component at W=11 (mcfm v2_G).

The paper states an eleven-frame window, so every row that HAS a window is
measured at eleven. Only the last row has one: frozen, +CA and +CA+SA have the
temporal component off, and with it off the conditioning is a single frame, so
those three rows are window-independent and are unchanged from the W=3 build.

Emitted for both object sets the artifact carries: 24 (batches A-C) and 42 (A-E).

PSNR/SSIM are IMAGE space, from each run's final_eval.json at the supervised view.
Only flicker/jerk/drift are texel space. The old header called the whole block
"texel-space"; that was wrong and is corrected in the page.
"""
import json, os, glob, statistics as st
from scipy.stats import binomtest

E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(E)

OBJ42 = json.load(open('out/fullrate_table_batchALL.json'))['objects']
OBJ24 = ['ancient_lady_effect_2','hand_rorschach','pumpkin_rot','spot_lava',
         'animal_blob_crack','animal_blob_orange_crack','chair_ice','chair_moss',
         'chair_real_wooden_crack','napolean_teapot_crack','napolean_waves',
         'octopus_rainbow','octopus_rust','octopus_sparkle','octopus_tar',
         'dragon_mush','dragon_mush2','fish_glitter','fish_ink','octocat_clay',
         'octocat_shine','sheep_mud','sheep_soil','tie_fighter_bw']
assert len(OBJ24) == 24 and set(OBJ24) <= set(OBJ42)

def arm(objs, tag, frozen=False):
    """Per-object metrics for one arm, in the given object order."""
    out = {}
    for o in objs:
        t = json.load(open('out/TEXEL/%s_%s.json' % (o, tag)))
        fe = json.load(open('runs/%s/final_eval.json' % t['run']))
        k = 'frozen' if frozen else 'final'
        out[o] = dict(psnr=fe[k]['psnr_mean'], ssim=fe[k]['ssim_mean'],
                      flicker=t['frozen_flicker' if frozen else 'texel_flicker'],
                      accel=t['frozen_jerk'    if frozen else 'texel_jerk'],
                      drift=t['frozen_drift'   if frozen else 'texel_drift'])
    return out

def sign(a, b, objs, key, lower_better):
    """Objects on which a beats b, plus a two-sided sign test."""
    w = sum((a[o][key] < b[o][key]) if lower_better else (a[o][key] > b[o][key]) for o in objs)
    return w, binomtest(w, len(objs), 0.5).pvalue

for objs, name in ((OBJ24, '24 objects (A-C)'), (OBJ42, '42 objects (A-E)')):
    # frozen comes free from the r19 pass; W=11 is v2_G, tag w11
    rows = [('Frozen TRELLIS.2',              arm(objs, 'r19', frozen=True)),
            ('+ LoRA CA',                     arm(objs, 'r19')),
            ('+ LoRA CA + LoRA SA',           arm(objs, 'r27')),
            ('+ LoRA CA + LoRA SA + Temporal', arm(objs, 'w11'))]
    print('\n=== %s   temporal component at W=11 (v2_G) ===' % name)
    print('%-34s %6s %7s %9s %9s %8s   %s' %
          ('Configuration','PSNR','SSIM','Flicker','Accel.','Drift','vs row above, flicker'))
    prev = None
    for lab, d in rows:
        m = {k: st.mean(d[o][k] for o in objs) for k in ('psnr','ssim','flicker','accel','drift')}
        sig = '-'
        if prev is not None:
            w, p = sign(d, prev, objs, 'flicker', True)
            sig = '%d/%d, p=%.2g' % (w, len(objs), p)
        print('%-34s %6.2f %7.4f %9.5f %9.5f %8.4f   %s' %
              (lab, m['psnr'], m['ssim'], m['flicker'], m['accel'], m['drift'], sig))
        prev = d
    # what each step buys
    print('  step deltas:')
    for i in (1, 2, 3):
        a, b = rows[i][1], rows[i-1][1]
        dp = st.mean(a[o]['psnr'] for o in objs) - st.mean(b[o]['psnr'] for o in objs)
        df = (st.mean(a[o]['flicker'] for o in objs)/st.mean(b[o]['flicker'] for o in objs)-1)*100
        da = (st.mean(a[o]['accel']  for o in objs)/st.mean(b[o]['accel']  for o in objs)-1)*100
        wp, pp = sign(a, b, objs, 'psnr', False)
        wf, pf = sign(a, b, objs, 'flicker', True)
        wa, pa = sign(a, b, objs, 'accel', True)
        print('    %-30s dPSNR %+6.2f dB (%d/%d, p=%.2g)  dFlicker %+6.1f%% (%d/%d, p=%.2g)  dAccel %+6.1f%% (%d/%d, p=%.2g)'
              % (rows[i][0], dp, wp, len(objs), pp, df, wf, len(objs), pf, da, wa, len(objs), pa))
