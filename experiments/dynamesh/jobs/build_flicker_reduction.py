"""build_flicker_reduction.py -- the flicker-reduction figure the paper promises.

The main paper says "we visualize the reduction in flicker achieved by DynaMesh"; the
supplement had no such figure. Nothing needs re-rendering: every
out/TEXEL/<obj>_<arm>.json already carries both sides of the comparison for the SAME
object -- frozen_flicker / texel_flicker and their per-frame curves -- because the texel
pass measures the frozen generator alongside the adapted one.

Two panels, from that one file per object:
  (a) 42 paired points, frozen -> ours. A mean hides whether the gain is universal; the
      pairing shows it holds object by object, which is the claim being made.
  (b) the per-frame curve, averaged over the 42, so the reduction is visible across the
      whole clip rather than only in the mean.

Both axes are log in (a): flicker spans an order of magnitude across objects, and on a
linear axis the 42 pairs collapse onto the bottom-left corner.
"""
import json, os, sys
import numpy as np

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
os.chdir(E)
ARM = sys.argv[1] if len(sys.argv) > 1 else 'w11'      # the shipped window
OBJS = [l.split('\t')[1] for l in open('jobs/rung37_objects.tsv').read().splitlines()[1:] if l.strip()]

fro, our, cur_f, cur_o = [], [], [], []
missing = []
for o in OBJS:
    p = f'out/TEXEL/{o}_{ARM}.json'
    if not os.path.exists(p):
        missing.append(o); continue
    d = json.load(open(p))
    fro.append(d['frozen_flicker']); our.append(d['texel_flicker'])
    # per-frame curves are per-object length (121 or 150); resample to a common axis so
    # objects of different clip length can be averaged without truncating the 150s.
    for src, dst in (('frozen_flicker_per_frame', cur_f), ('flicker_per_frame', cur_o)):
        v = np.asarray(d[src], dtype=float)
        dst.append(np.interp(np.linspace(0, 1, 151), np.linspace(0, 1, len(v)), v))
assert not missing, f'missing texel json for {missing}'
fro, our = np.asarray(fro), np.asarray(our)
# NORMALIZED, not a frame index. Clips are 121 or 150 frames; a frame axis has to be
# truncated to the shortest, which shows 120 -- a number that is neither real length and
# reads as an error. Resampling onto [0,1] keeps every clip's full extent.
cf = np.mean(cur_f, 0)
co = np.mean(cur_o, 0)
red = 100 * (1 - our / fro)
won = int((our < fro).sum())
print(f'{ARM}: {len(fro)} objects   frozen {fro.mean():.5f} -> ours {our.mean():.5f}   '
      f'mean reduction {red.mean():.1f}%   ours lower on {won}/{len(fro)}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OURS  = '#2f6fa8'      # blue
FROZ  = '#c0504d'      # red, frozen TRELLIS.2
GREEN = '#2e7d32'      # green, ours, in the per-frame plot

os.makedirs('out/FLICKER', exist_ok=True)

# ---- figure B: per-frame curves, frame index, no fill ----------------------------
figB, ax = plt.subplots(figsize=(4.6, 3.0))
x = np.linspace(0, 1, cf.size)
ax.plot(x, cf, color=FROZ,  lw=1.9, label='Frozen TRELLIS.2')
ax.plot(x, co, color=GREEN, lw=1.9, label='DynaMesh (ours)')
ax.set_xlabel('Normalized time step'); ax.set_ylabel('Flicker')
# 0 and 1 must both be drawn: matplotlib's default locator dropped them and the axis
# started at 0.0 without a tick, which read as a missing origin.
ax.set_xlim(0, 1); ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_ylim(0, max(cf.max(), co.max()) * 1.15)
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
figB.tight_layout(); figB.patch.set_facecolor('white'); ax.set_facecolor('white')
for ext in ('pdf', 'png', 'svg'):
    figB.savefig(f'out/FLICKER/dynamesh_flicker_perframe.{ext}', dpi=200,
                 facecolor='white', edgecolor='none')

json.dump(dict(arm=ARM, n=len(fro), frozen_mean=float(fro.mean()), ours_mean=float(our.mean()),
               mean_reduction_pct=float(red.mean()), median_reduction_pct=float(np.median(red)),
               objects_lower=won,
               per_object=[dict(obj=o, frozen=float(f), ours=float(u))
                           for o, f, u in zip(OBJS, fro, our)]),
          open('out/FLICKER/flicker_reduction.json', 'w'), indent=1)
print('wrote out/FLICKER/dynamesh_flicker_perframe.{pdf,png,svg}')
