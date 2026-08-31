"""build_judge_table.py -- aggregate llm_judge.py output into the supplementary table.

Reads the JSONL written by llm_judge.py and reports, per method:
  effect match / temporal smoothness / geometry preservation / surface adherence,
each averaged over repeats first, then over views, then over objects. Averaging in
that order keeps an object with a missing view from silently outweighing the rest.

Also prints the two things that decide whether the numbers are usable at all:
  CALIBRATION  gt-vs-gt on effect_match must sit at 5.0. It is the judge grading the
               reference against itself; anything below ~4.8 means the rubric is not
               landing and the other rows should not be read.
  NOISE        mean within-video std across repeats. This is the number that says
               whether 3 repeats was enough, and it belongs in the caption.

Usage
    python3 jobs/build_judge_table.py --in out/JUDGE/pilot.jsonl [--tex judge_table.tex]
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path

VIEWS = ['train', 'diagA', 'diagB', 'diagC']
QS = ['effect_match', 'temporal_smoothness', 'geometry_preservation',
      'surface_adherence']
QLAB = {'effect_match': 'Effect match', 'temporal_smoothness': 'Temporal smooth.',
        'geometry_preservation': 'Geometry pres.', 'surface_adherence': 'Surface adher.'}
ORDER = ['frozen', 'meshnca', 'sv4d2', 'dg4d', 'l4gm', 'ours']
NAME = {'frozen': 'Frozen TRELLIS.2', 'meshnca': 'MeshNCA', 'l4gm': 'L4GM',
        'sv4d2': 'SV4D 2.0', 'dg4d': 'DreamGaussian4D', 'ours': 'Ours', 'gt': 'GT'}
# TWO methods are frame-capped, not one. Counted off the source PNG directories that
# build_per_object_pages.py encodes from, not guessed: SV4D2 emits 21 frames and DG4D
# emits 32, where gt/frozen/meshnca/l4gm/ours all emit 150 over the same 6 seconds
# (pumpkin_rot is 121 throughout, being a 121-frame clip). Both caps depress
# temporal_smoothness for a reason that is the method's own output length rather than
# its quality, so both are printed as a column instead of being hidden.
NFRAMES = {'frozen': 150, 'meshnca': 150, 'l4gm': 150, 'sv4d2': 21, 'dg4d': 32,
           'ours': 150, 'gt': 150}
CAPPED = {'sv4d2', 'dg4d'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--tex', default='')
    a = p.parse_args()

    recs = [json.loads(l) for l in Path(a.inp).read_text().splitlines() if l.strip()]
    objs = sorted({r['obj'] for r in recs})
    print(f'{len(recs)} judgements, {len(objs)} objects: {", ".join(objs)}')
    print(f'{sum(r.get("tokens") or 0 for r in recs)/1e6:.2f}M tokens\n')

    # repeats -> one value per (obj, method, view, question)
    cell = collections.defaultdict(list)
    for r in recs:
        cell[(r['obj'], r['method'], r['view'], r['question'])].append(r['score'])

    noise = [st.stdev(v) for v in cell.values() if len(v) > 1]
    reps = collections.Counter(len(v) for v in cell.values())
    print(f'repeats per cell: {dict(sorted(reps.items()))}')
    if noise:
        print(f'within-video std across repeats: mean {st.mean(noise):.3f}, '
              f'max {max(noise):.3f}, {sum(n == 0 for n in noise)}/{len(noise)} '
              f'cells unanimous')

    # Coverage. A cell can go missing because its source mp4 is undecodable and the
    # API rejects it with a 400, which is exactly what happened to eight MeshNCA
    # training-view clips. Averaging over what survived would hide that, so the
    # shortfall is printed per method and view before any score is.
    gaps = collections.Counter()
    for o in objs:
        for m in ORDER + ['gt']:
            for v in (['train'] if m == 'gt' else VIEWS):
                for q in QS:
                    if q == 'effect_match' and v != 'train':
                        continue
                    if (o, m, v, q) not in cell:
                        gaps[(m, v)] += 1
    if gaps:
        print('ABSENT cells (source video undecodable, or never judged):')
        for (m, v), n in sorted(gaps.items(), key=lambda kv: -kv[1]):
            print(f'    {NAME[m]:18s} {v:6s} {n:>4d}')
    else:
        print('coverage: every cell present')

    # A cell can be present but under-replicated: a repeat that 429'd leaves the other
    # two behind, and averaging three-repeat cells against one-repeat cells silently
    # mixes precisions. Reported separately, because "every cell present" is NOT the
    # same claim as "every cell has its three repeats".
    thin = collections.Counter()
    for (o, m, v, q), vals in cell.items():
        if len(vals) < 3:
            thin[m] += 3 - len(vals)
    if thin:
        short = sum(1 for v in cell.values() if len(v) < 3)
        print(f'UNDER-REPLICATED: {short} of {len(cell)} cells have fewer than 3 '
              f'repeats ({sum(thin.values())} judgements short)')
        print('    ' + '  '.join(f'{NAME[m]}:{n}' for m, n in
                                 sorted(thin.items(), key=lambda kv: -kv[1])))

    calib = [v for k, v in cell.items() if k[1] == 'gt' and k[3] == 'effect_match']
    if calib:
        flat = [s for v in calib for s in v]
        print(f'CALIBRATION gt-vs-gt effect_match: {st.mean(flat):.2f} '
              f'over {len(flat)} judgements  '
              f'{"OK" if st.mean(flat) >= 4.8 else "<<< RUBRIC NOT LANDING"}')
    print()

    # view -> object -> method mean
    def score(method, q):
        per_obj = []
        for o in objs:
            vals = [st.mean(v) for k, v in cell.items()
                    if k[0] == o and k[1] == method and k[3] == q]
            if vals:
                per_obj.append(st.mean(vals))
        return (st.mean(per_obj), len(per_obj)) if per_obj else (None, 0)

    hdr = f'{"method":18s}{"n":>4s}' + ''.join(f'{QLAB[q]:>18s}' for q in QS)
    print(hdr)
    print('-' * len(hdr))
    rows = {}
    for m in ORDER + ['gt']:
        cells = {q: score(m, q) for q in QS}
        if all(c[0] is None for c in cells.values()):
            continue
        rows[m] = cells
        n = max(c[1] for c in cells.values())
        line = f'{NAME[m]:18s}{n:>4d}'
        for q in QS:
            v = cells[q][0]
            line += f'{v:>18.2f}' if v is not None else f'{"--":>18s}'
        print(line)

    if a.tex:
        best = {q: max((rows[m][q][0], m) for m in rows
                       if m != 'gt' and rows[m][q][0] is not None)[1] for q in QS}
        L = [r'% Generated by jobs/build_judge_table.py -- do not hand-edit.',
             r'\begin{table}[t]', r'\centering', r'\small',
             r'\begin{tabular}{lc' + 'c' * len(QS) + '}', r'\toprule',
             'Method & Frames & ' + ' & '.join(QLAB[q] + r'\,$\uparrow$' for q in QS)
             + r' \\', r'\midrule']
        for m in ORDER:
            if m not in rows:
                continue
            cs = []
            for q in QS:
                v = rows[m][q][0]
                s = '--' if v is None else f'{v:.2f}'
                cs.append(r'\textbf{' + s + '}' if best.get(q) == m and v else s)
            L.append(f'{NAME[m]} & {NFRAMES[m]} & ' + ' & '.join(cs) + r' \\')
        if 'gt' in rows:
            L.append(r'\midrule')
            cs = [('--' if rows['gt'][q][0] is None else f'{rows["gt"][q][0]:.2f}')
                  for q in QS]
            L.append(r'\emph{Driving video} & 150 & ' + ' & '.join(cs) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}',
              r'\caption{\textbf{Semantic evaluation with an LLM judge.} '
              r'Gemini rates each rendered video 1--5 per question; every video is '
              r'judged three times and averaged. Scores average over repeats, then '
              r'the training view plus three unseen views, then objects. '
              r'\emph{Effect match} is scored at the training view against the '
              r'driving video itself rather than a text description. '
              r'\textbf{SV4D 2.0 and DreamGaussian4D emit 21 and 32 frames '
              r'respectively where every other method emits 150 over the same '
              r'duration}, which depresses their temporal scores for a reason that is '
              r'each method\textquotesingle s own output length rather than its '
              r'quality; the frame count is reported as a column rather than hidden.}',
              r'\label{tab:judge}', r'\end{table}']
        Path(a.tex).write_text('\n'.join(L) + '\n')
        print(f'\nwrote {a.tex}')


if __name__ == '__main__':
    main()
