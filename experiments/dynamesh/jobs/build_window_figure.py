#!/usr/bin/env python3
"""Window-size ablation: table + graph for the supplementary.

Reads everything from disk. Runs are resolved from each run's own config.json,
never from a glob over run-directory names, because several objects carry more
than one 30-epoch run per arm and a glob would pick a different checkpoint than
the one the texel metrics were computed on. The texel json's "run" field is the
authority; we check it resolves back to the same object and mcfm mode.

W=1 is the no-blend rung27 row. It and W=3 are, by construction, the last two
rows of Table S2, so the sweep extends that table rather than restating it.
"""
import json, os, glob, sys
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# width -> (mcfm mode in config.json, tag used in out/TEXEL/<obj>_<tag>.json)
ARMS = [(1, None, 'r27'), (3, 'v2_D', 'r27mcfm'), (5, 'v2_E', 'w5'),
        (7, 'v2_F', 'w7'), (11, 'v2_G', 'w11'), (13, 'v2_H', 'w13'),
        (15, 'v2_I', 'w15')]

runs = {}
for cfgp in glob.glob('runs/*/config.json'):
    try:
        c = json.load(open(cfgp))
    except Exception:
        continue
    p = (c.get('mesh') or '').split('/')
    runs[os.path.basename(os.path.dirname(cfgp))] = dict(
        obj=p[p.index('data') + 1] if 'data' in p else None,
        rung=c.get('rung'), mcfm=c.get('mcfm'), ep=c.get('epochs'),
        seed=c.get('seed'), tg=c.get('targets'), rank=c.get('rank'))

def is_arm(r, mode):
    return (r['rung'] == 27 and r['tg'] == 'qkvo+sa' and r['rank'] == 4
            and r['seed'] == 42 and r['ep'] == 30 and r['mcfm'] == mode)

OBJS = sorted({r['obj'] for r in runs.values() if is_arm(r, 'v2_G')})
assert len(OBJS) == 42, len(OBJS)

per = {}   # width -> metric -> per-object array, object order = OBJS
for w, mode, tag in ARMS:
    fl, jk, dr, ps, ss = [], [], [], [], []
    for o in OBJS:
        t = json.load(open('out/TEXEL/%s_%s.json' % (o, tag)))
        rn = t['run']
        assert rn in runs and runs[rn]['obj'] == o and runs[rn]['mcfm'] == mode, (o, w, rn)
        fl.append(t['texel_flicker']); jk.append(t['texel_jerk']); dr.append(t['texel_drift'])
        fe = json.load(open('runs/%s/final_eval.json' % rn))['final']
        ps.append(fe['psnr_mean']); ss.append(fe['ssim_mean'])
    per[w] = dict(flicker=np.array(fl), accel=np.array(jk), drift=np.array(dr),
                  psnr=np.array(ps), ssim=np.array(ss))

W = [w for w, _, _ in ARMS]

# ---------------------------------------------------------------- json
out = dict(n_objects=len(OBJS), objects=OBJS, rows=[])
for w in W:
    out['rows'].append(dict(window=w, **{k: float(v.mean()) for k, v in per[w].items()}))
json.dump(out, open('out/window_sweep.json', 'w'), indent=1)

# ------------------------------------------------- paired tests vs W=3
ref = 3
print('paired Wilcoxon against W=%d, n=42' % ref)
for w in W:
    if w == ref:
        continue
    line = 'W=%-3d' % w
    for m, better in (('flicker', 'lt'), ('psnr', 'gt')):
        a, b = per[w][m], per[ref][m]
        wins = int((a < b).sum()) if better == 'lt' else int((a > b).sum())
        p = wilcoxon(a, b).pvalue
        line += '  %s %2d/42 p=%.2g' % (m, wins, p)
    print(line)

# ---------------------------------------------------------------- tex
with open('out/window_sweep.tex', 'w') as f:
    f.write('\\begin{tabular}{l cc cc}\n\\toprule\n')
    f.write(' & \\multicolumn{2}{c}{Reconstruction} & \\multicolumn{2}{c}{Temporal} \\\\\n')
    f.write('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n')
    f.write('Window $|\\mathcal{W}|$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ '
            '& Flicker $\\downarrow$ & Accel. $\\downarrow$ \\\\\n\\midrule\n')
    for w in W:
        r = {k: v.mean() for k, v in per[w].items()}
        lab = '1 (no blending)' if w == 1 else str(w)
        f.write('%-16s & %.2f & %.4f & %.5f & %.5f \\\\\n'
                % (lab, r['psnr'], r['ssim'], r['flicker'], r['accel']))
    f.write('\\bottomrule\n\\end{tabular}\n')

# --------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FL = '#1f4e79'; PS = '#c0504d'
fig, ax = plt.subplots(figsize=(5.0, 3.0))
fl = [per[w]['flicker'].mean() for w in W]
ps = [per[w]['psnr'].mean() for w in W]

# No error bars. The spread across the 42 objects is between-object variance and
# swamps a 0.07 dB range; the paired tests in the caption are the honest statement.
# The right axis spans 1.6 dB so a flat PSNR curve looks flat.
ax.plot(W, fl, color=FL, marker='o', ms=5, lw=1.9, label='Flicker', zorder=3)
ax.set_xlabel('Temporal window size $|\\mathcal{W}|$')
ax.set_ylabel('Flicker $\\downarrow$', color=FL)
ax.tick_params(axis='y', labelcolor=FL)
ax.set_xticks(W); ax.set_xticklabels([str(w) for w in W])
ax.set_ylim(0.0045, 0.0075)
ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)

ax2 = ax.twinx()
ax2.plot(W, ps, color=PS, marker='s', ms=5, lw=1.9, ls='--', label='PSNR', zorder=3)
ax2.set_ylabel('PSNR (dB) $\\uparrow$', color=PS)
ax2.tick_params(axis='y', labelcolor=PS)
ax2.set_ylim(24.0, 25.6)

ax.axvline(3, color='0.55', lw=.9, ls=':', zorder=1)
ax.annotate('ours', xy=(3, 0.00577), xytext=(3.5, 0.00655), fontsize=9,
            color='0.35', arrowprops=dict(arrowstyle='-', color='0.55', lw=.8))

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper center', ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig('out/window_sweep.%s' % ext, dpi=200)
print('\nwrote out/window_sweep.{json,tex,pdf,png}')
