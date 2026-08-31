#!/usr/bin/env python3
"""Components off/on ablation, evaluated in VIDEO space.

The pixel twin of the texel ladder (paper Table S2). Four rows, one component
switched on at a time, over the same 42 objects as Table 1.

CONVENTION MATCHES TABLE 1, and must stay matched or the bottom row stops being
comparable to the headline number: flicker/accel/drift are averaged over the three
UNSEEN views (diagA/diagB/diagC), while PSNR/SSIM are taken at the supervised view
only, because the driving clip is a single camera and the unseen views have nothing
to score against.

REFERENCE. Only the 2D-copy dirs are read. out/FULLRATE and out/FULLRATE_W11 are the
raw-frame reference and read ~6 dB low (19.08 vs 24.90 on the same runs); mixing them
in is exactly how the superseded W=11 comparison happened.

FROZEN ROW comes from FULLRATE_R27's frozen panel. Inside an --mcfm run the frozen
arm receives the blended conditioning too, so the frozen half of a 27m/27g render is
frozen+MCFM, not frozen. Arm 27 has MCFM off, so its left panel is the real thing.
"""
import json, os, statistics as st

E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(E)
U = ['diagA', 'diagB', 'diagC']

OBJS = json.load(open('out/fullrate_table_batchALL.json'))['objects']

# label : (dir, which panel)      -- all four dirs are the 2D-copy reference
# The temporal row is W=11, the window the main text and Fig. 4 state. W=3 is carried
# as a secondary row because Table S1 currently states [t-1,t,t+1]; both round to
# Table 1's printed 24.9 / 0.798, so either can be the method row, but only one should be.
ROWS = [('Frozen TRELLIS.2',                 'FULLRATE_R27',    'frozen'),
        ('+ LoRA CA',                        'FULLRATE_R19',    'ours'),
        ('+ LoRA CA + LoRA SA',              'FULLRATE_R27',    'ours'),
        ('+ LoRA CA + LoRA SA + Temporal',   'FULLRATE_W11CG',  'ours'),
        ('   the same at W=3',               'FULLRATE_CG',     'ours')]

def load(d, panel):
    per = {}
    for o in OBJS:
        p = 'out/%s/%s.json' % (d, o)
        if not os.path.exists(p):
            continue
        f = json.load(open(p))[o]['full']
        sup = f.get('train|%s' % panel)
        uns = [f['%s|%s' % (v, panel)] for v in U if '%s|%s' % (v, panel) in f]
        if sup is None or len(uns) != 3:
            continue
        per[o] = dict(psnr=sup.get('psnr'), ssim=sup.get('ssim'),
                      flicker=st.mean(x['flicker'] for x in uns),
                      accel=st.mean(x['accel'] for x in uns),
                      drift=st.mean(x['drift'] for x in uns))
    return per

data = {lab: load(d, p) for lab, d, p in ROWS}
common = set(OBJS)
for lab, _, _ in ROWS[:4]:          # the W=11 extra row does not gate the ladder
    common &= set(data[lab])
common = sorted(common)
print('objects on every ladder row: %d of %d' % (len(common), len(OBJS)))

out = dict(n_objects=len(common), objects=common, views_temporal=U,
           view_fidelity='train', reference='2D-copy', rows=[])
print('\n%-34s %6s %7s %9s %9s %8s   n' %
      ('Configuration', 'PSNR', 'SSIM', 'Flicker', 'Accel.', 'Drift'))
for lab, d, panel in ROWS:
    per = data[lab]
    objs = [o for o in common if o in per]
    m = {k: st.mean(per[o][k] for o in objs) for k in
         ('psnr', 'ssim', 'flicker', 'accel', 'drift')}
    out['rows'].append(dict(label=lab.strip(), dir=d, panel=panel,
                            n=len(objs), **{k: float(v) for k, v in m.items()}))
    print('%-34s %6.2f %7.4f %9.5f %9.5f %8.4f  %2d' %
          (lab, m['psnr'], m['ssim'], m['flicker'], m['accel'], m['drift'], len(objs)))

json.dump(out, open('out/video_ladder.json', 'w'), indent=1)
print('\nwrote out/video_ladder.json')

# The check that makes this table worth printing: the all-components row has to be
# Table 1's number. If this drifts, something changed reference or view set.
b = out['rows'][3]
# Compared at the precision Table 1 actually prints (24.9 / 0.798), not at full
# precision: W=3 gives 24.90 and W=11 gives 24.89, and both are the same printed row.
print('all-components row: %.2f / %.4f  ->  prints as %.1f / %.3f   vs Table 1: 24.9 / 0.798   %s' %
      (b['psnr'], b['ssim'], b['psnr'], b['ssim'],
       'MATCH' if round(b['psnr'], 1) == 24.9 and round(b['ssim'], 3) == 0.798
       else 'MISMATCH -- do not publish'))
