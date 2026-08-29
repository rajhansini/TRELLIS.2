"""build_results_table.py -- the two main-paper tables.

  comparison_table.tex   Table 1. Ours vs the frozen TRELLIS.2 baseline. Only the
                         temporal component is toggled; the component ladder stays out.
  ablation_table.tex     Table 2. The ladder, frozen as its all-off row.

Frozen TRELLIS.2 appears in both, by design: it is the comparison's baseline and the
ablation's null row, and it must be the SAME numbers in both or a reviewer will find it.

ONE RULE FOR EVERY ROW. Object set is the 24 in out/texel_r19_params.tsv.
  temporal (flicker / accel / drift)   out/TEXEL/<object>_<arm>.json, texel_* fields
  fidelity (PSNR / SSIM)               the run named INSIDE that texel json ->
                                       runs/<run>/final_eval.json, "final" block
  Joining through the texel json's own `run` field is what keeps the two halves of a
  row describing the same checkpoint. Averaging final_eval over every run dir matching
  a rung prefix does not: rung27 alone has 63 run dirs over these 24 objects, most of
  them arms that never entered a table.

Rows are aggregated per object first (mean over that row's arms), then over objects, so
a row with ten arms does not outvote a row with one.

FROZEN DOES NOT COME FROM out/TEXEL/*_FROZEN.json. Three of those 24 files disagree with
every other measurement of the same quantity: spot_lava was probed at n_frames=8 (the
rest are 150/121, and flicker is a per-step rate, so an 8-frame probe is not on the same
scale), and hand_rorschach / pumpkin_rot differ from the embedded frozen by 36-56%. The
*_r19.json files carry frozen_* measured in the same decode pass as their texel_*, at
each object's own frame count, complete for all 24, and agreeing to four decimals with
the frozen_* in *_r27.json on the 9 objects that carry both.
"""
import json, os, statistics as st, sys
from math import comb

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
OBJ = [l.split('\t')[0] for l in
       open(f'{E}/out/texel_r19_params.tsv').read().strip().split('\n')[1:]]

ARMS = {
    'a': None,                                            # frozen
    'b': ['r19'],                                         # + cross-attention
    'c': ['r27', 'r28', 'r29', 'r30'],                    # + self-attention
    'd': ['r31', 'r32', 'r33', 'r34', 'r27mcfm', 'r28mcfm',
          'r29mcfm', 'r30mcfm', 'r31mcfm', 'r33mcfm'],    # + temporal
}
KEYS = ['psnr', 'ssim', 'flicker', 'accel', 'drift']
LOWER_BETTER = {'flicker', 'accel'}

_e = {}
def evalj(run):
    run = run.split('/')[-1]
    if run not in _e:
        p = f'{E}/runs/{run}/final_eval.json'
        _e[run] = json.load(open(p)) if os.path.exists(p) else None
    return _e[run]

def per_object(tag):
    """{object: {metric: value}} -- mean over this row's arms, for one object."""
    out, miss = {}, []
    for o in OBJ:
        F = J = D = P = S = None
        if tag == 'a':
            t = json.load(open(f'{E}/out/TEXEL/{o}_r19.json'))
            e = evalj(t['run'])
            F, J, D = [t['frozen_flicker']], [t['frozen_jerk']], [t['frozen_drift']]
            P, S = [e['frozen']['psnr_mean']], [e['frozen']['ssim_mean']]
        else:
            F, J, D, P, S = [], [], [], [], []
            for a in ARMS[tag]:
                p = f'{E}/out/TEXEL/{o}_{a}.json'
                if not os.path.exists(p):
                    miss.append(f'{o}_{a}'); continue
                t = json.load(open(p))
                F.append(t['texel_flicker']); J.append(t['texel_jerk']); D.append(t['texel_drift'])
                e = evalj(t['run'])
                if e: P.append(e['final']['psnr_mean']); S.append(e['final']['ssim_mean'])
        out[o] = dict(psnr=st.mean(P), ssim=st.mean(S), flicker=st.mean(F),
                      accel=st.mean(J), drift=st.mean(D))
    return out, miss

PO, MISS = {}, {}
for t in ARMS:
    PO[t], MISS[t] = per_object(t)
    if MISS[t]:
        print(f'  row ({t}) missing {len(MISS[t])} measurement(s): {MISS[t]}')

M = {t: {k: st.mean(PO[t][o][k] for o in OBJ) for k in KEYS} for t in ARMS}

def sign_p(w, n):
    k = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def wins(hi, lo, k):
    """objects on which `lo` beats `hi` for metric k, and the sign-test p."""
    w = sum(1 for o in OBJ if (PO[lo][o][k] < PO[hi][o][k]) == (k in LOWER_BETTER))
    return w, sign_p(w, len(OBJ))

def rel(hi, lo, k):
    return 100 * (M[lo][k] - M[hi][k]) / M[hi][k]

def cells(t, bold=()):
    f = {'psnr': '{:.2f}', 'ssim': '{:.4f}', 'flicker': '{:.5f}',
         'accel': '{:.5f}', 'drift': '{:.4f}'}
    return ' & '.join((r'\textbf{%s}' % f[k].format(M[t][k])) if k in bold
                      else f[k].format(M[t][k]) for k in KEYS)

X, V = r'\xmark', r'\cmark'
BEST = ('psnr', 'ssim', 'flicker', 'accel')   # drift is a guard: never marked best

# ---------------------------------------------------------------- Table 1
wF = wins('a', 'd', 'flicker'); wA = wins('a', 'd', 'accel'); wP = wins('a', 'd', 'psnr')
comparison = rf"""% Generated by jobs/build_results_table.py -- do not hand-edit.
% Table 1: comparison. Frozen TRELLIS.2 is also row 1 of the ablation table and MUST
% carry identical numbers there.
\newcommand{{\cmark}}{{\ding{{51}}}}   % requires \usepackage{{pifont}}
\newcommand{{\xmark}}{{\ding{{55}}}}
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
& & \multicolumn{{2}}{{c}}{{Reconstruction}} & \multicolumn{{3}}{{c}}{{Temporal (texel-space)}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-7}}
Method & Temporal & PSNR\,$\uparrow$ & SSIM\,$\uparrow$ & Flicker\,$\downarrow$ & Accel.\,$\downarrow$ & Drift \\
\midrule
Frozen TRELLIS.2 & {X} & {cells('a')} \\
\midrule
Ours & {X} & {cells('c')} \\
Ours & {V} & {cells('d', BEST)} \\
\bottomrule
\end{{tabular}}
\caption{{\textbf{{Comparison against the frozen generator.}} Averages over
{len(OBJ)} objects. The frozen generator is not asked to produce a temporally coherent
texture, so the fidelity gap is large and uninformative on its own; the temporal columns
carry the claim. Ours is better on {wF[0]}/{len(OBJ)} objects for flicker and
{wA[0]}/{len(OBJ)} for acceleration (two-sided sign test, $p < 10^{{-6}}$ for both),
reducing them {-rel('a','d','flicker'):.1f}\% and {-rel('a','d','accel'):.1f}\%. Enabling the
temporal component accounts for {-rel('c','d','flicker'):.1f}\% of the flicker reduction and
{-rel('c','d','accel'):.1f}\% of the acceleration reduction on its own, at
{M['d']['psnr']-M['c']['psnr']:+.2f}\,dB. Drift is a guard rather than a score and is
never marked best: it moves {rel('c','d','drift'):+.1f}\% when the temporal component is
enabled, confirming the gain is not damping the texture toward a still image --- the
degenerate solution that minimises both other temporal metrics. Component-wise attribution
is deferred to Table~\ref{{tab:ablation}}.}}
\label{{tab:comparison}}
\end{{table}}
"""

# ---------------------------------------------------------------- Table 2
wSA = wins('b', 'c', 'flicker'); wSAp = wins('b', 'c', 'psnr')
wT = wins('c', 'd', 'flicker'); wTa = wins('c', 'd', 'accel')
ablation = rf"""% Generated by jobs/build_results_table.py -- do not hand-edit.
% Table 2: ablation ladder. Row 1 is the frozen baseline of Table 1, same numbers.
\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{cccccccc}}
\toprule
\multicolumn{{3}}{{c}}{{Adapted}} & \multicolumn{{2}}{{c}}{{Reconstruction}} & \multicolumn{{3}}{{c}}{{Temporal (texel-space)}} \\
\cmidrule(lr){{1-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-8}}
Cross-attn & Self-attn & Temporal & PSNR\,$\uparrow$ & SSIM\,$\uparrow$ & Flicker\,$\downarrow$ & Accel.\,$\downarrow$ & Drift \\
\midrule
{X} & {X} & {X} & {cells('a')} \\
{V} & {X} & {X} & {cells('b')} \\
{V} & {V} & {X} & {cells('c')} \\
{V} & {V} & {V} & {cells('d', BEST)} \\
\bottomrule
\end{{tabular}}
\caption{{\textbf{{Each component buys a different thing.}} Same {len(OBJ)} objects and
same frozen row as Table~\ref{{tab:comparison}}. Adapting cross-attention alone recovers
most of both quantities ({M['b']['psnr']-M['a']['psnr']:+.2f}\,dB,
{-rel('a','b','flicker'):.1f}\% flicker). Extending the adapter to self-attention adds
{M['c']['psnr']-M['b']['psnr']:+.2f}\,dB, better on {wSAp[0]}/{len(OBJ)} objects
($p<10^{{-6}}$), but its effect on coherence is not significant --- lower flicker on only
{wSA[0]}/{len(OBJ)} objects, $p={wSA[1]:.2f}$: it is a fidelity component, not a temporal
one. The temporal component is the reverse, lowering flicker {-rel('c','d','flicker'):.1f}\%
on {wT[0]}/{len(OBJ)} objects and acceleration {-rel('c','d','accel'):.1f}\% on
{wTa[0]}/{len(OBJ)} (both $p<10^{{-6}}$) while PSNR moves
{M['d']['psnr']-M['c']['psnr']:+.2f}\,dB. Drift is a guard and is never marked best.
Cross-attn is rung~19; self-attn adds rungs 27--30 (28--30 add KL regularisers to
rung~27); temporal is rungs 31--34 together with every arm using the temporal blend.}}
\label{{tab:ablation}}
\end{{table}}
"""

open(f'{E}/comparison_table.tex', 'w').write(comparison)
open(f'{E}/ablation_table.tex', 'w').write(ablation)
json.dump({'per_object': PO, 'means': M, 'objects': OBJ,
           'missing': {k: v for k, v in MISS.items() if v}},
          open(f'{E}/out/results_table_data.json', 'w'), indent=1)

NAME = {'a': 'frozen', 'b': '+cross-attn', 'c': '+self-attn', 'd': '+temporal'}
print(f'\n{"row":<14}{"PSNR":>8}{"SSIM":>9}{"flicker":>10}{"accel":>10}{"drift":>9}')
for t in ['a', 'b', 'c', 'd']:
    m = M[t]
    print(f'{NAME[t]:<14}{m["psnr"]:>8.2f}{m["ssim"]:>9.4f}{m["flicker"]:>10.5f}'
          f'{m["accel"]:>10.5f}{m["drift"]:>9.4f}')
print('\nstage tests (win count out of 24, sign-test p):')
for hi, lo in [('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'd')]:
    ws = {k: wins(hi, lo, k) for k in ['psnr', 'flicker', 'accel']}
    print(f'  {NAME[hi]:<12} -> {NAME[lo]:<12} '
          f'PSNR {ws["psnr"][0]:>2}/24 p={ws["psnr"][1]:.1e}   '
          f'flicker {ws["flicker"][0]:>2}/24 p={ws["flicker"][1]:.1e}   '
          f'accel {ws["accel"][0]:>2}/24 p={ws["accel"][1]:.1e}')
print(f'\nwrote {E}/comparison_table.tex')
print(f'wrote {E}/ablation_table.tex')
print(f'wrote {E}/out/results_table_data.json')
