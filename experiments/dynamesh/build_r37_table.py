"""build_r37_table.py -- data for the rung37 vs rung27+MCFM comparison.

Same rules as the rung31 table so the two are directly comparable:
  temporal metrics  out/TEXEL/<obj>_<arm>.json   (texel_flicker / texel_jerk / texel_drift)
  PSNR + SSIM       final_eval.json of the run each texel json NAMES, never a re-derived path
  frozen row        the frozen_* fields inside <obj>_r19.json, NOT out/TEXEL/*_FROZEN.json
                    (three of those files are known bad)
  object set        strict intersection across every arm being reported

rung37 is the learned form of MCFM's own operator: same per-position 3-frame window,
trained Q/K instead of a fixed softmax, so it initialises AT the blend and can only
move away from it. That makes this a sharper test than rung31, which used a different
(joint spatio-temporal) geometry.
"""
import json, math, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
TX = E / 'out' / 'TEXEL'
ARMS = {'r19': '_r19', 'r27': '_r27', 'r27mcfm': '_r27mcfm', 'r37': '_r37'}


def objs_for(suf):
    return {p.name[:-len(suf) - 5] for p in TX.glob(f'*{suf}.json')}


def load(obj, suf):
    return json.loads((TX / f'{obj}{suf}.json').read_text())


def recon(texel):
    """PSNR/SSIM from the run the texel json names."""
    fe = E / 'runs' / texel['run'] / 'final_eval.json'
    if not fe.exists():
        return None, None
    d = json.loads(fe.read_text())
    return d['final']['psnr_mean'], d['final']['ssim_mean']


def signtest(a, b):
    """Two-sided exact binomial on wins, ties dropped. Returns (wins_b, n, p)."""
    wins = sum(1 for x, y in zip(a, b) if y < x)
    n = sum(1 for x, y in zip(a, b) if y != x)
    if n == 0:
        return 0, 0, 1.0
    C = math.comb
    tail = sum(C(n, k) for k in range(0, min(wins, n - wins) + 1)) / 2 ** n
    return wins, n, min(1.0, 2 * tail)


def main():
    sets = {a: objs_for(s) for a, s in ARMS.items()}
    for a, s in sets.items():
        print(f'  {a:9s} {len(s):3d} objects', file=sys.stderr)
    common = sorted(set.intersection(*sets.values()))
    print(f'  COMMON    {len(common):3d}', file=sys.stderr)
    if not common:
        sys.exit('no common objects yet')

    rows, arms_agg = [], {a: {'psnr': [], 'ssim': [], 'flicker': [], 'jerk': [], 'drift': []}
                          for a in list(ARMS) + ['frozen']}
    for o in common:
        rec = {'object': o}
        for a, suf in ARMS.items():
            t = load(o, suf)
            p, s = recon(t)
            rec[a] = dict(psnr=p, ssim=s, flicker=t['texel_flicker'],
                          jerk=t['texel_jerk'], drift=t['texel_drift'])
            for k in ('flicker', 'jerk', 'drift'):
                arms_agg[a][k].append(rec[a][k])
            if p is not None:
                arms_agg[a]['psnr'].append(p); arms_agg[a]['ssim'].append(s)
        t19 = load(o, '_r19')
        fr = dict(flicker=t19['frozen_flicker'], jerk=t19['frozen_jerk'], drift=t19['frozen_drift'])
        rec['frozen'] = fr
        for k in ('flicker', 'jerk', 'drift'):
            arms_agg['frozen'][k].append(fr[k])
        rows.append(rec)

    mean = {a: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
            for a, d in arms_agg.items()}

    tests = {}
    for k, better_low in (('psnr', False), ('ssim', False),
                          ('flicker', True), ('jerk', True), ('drift', True)):
        A = [r['r27mcfm'][k] for r in rows]
        B = [r['r37'][k] for r in rows]
        if better_low:
            w, n, p = signtest(A, B)
        else:                                  # higher is better: flip the comparison
            w, n, p = signtest([-x for x in A], [-x for x in B])
        tests[k] = dict(r37_wins=w, n=n, p=p)

    out = dict(n_objects=len(common), objects=common, rows=rows, mean=mean, signtest=tests)
    (E / 'out' / 'r37_table_data.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'n': len(common),
                      'mean': {a: {k: (round(v, 5) if v else v) for k, v in d.items()}
                               for a, d in mean.items()},
                      'signtest': tests}, indent=1))


if __name__ == '__main__':
    main()
