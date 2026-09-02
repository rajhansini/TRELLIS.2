"""report_window_timing.py -- inference cost against MCFM window width.

Reads out/timing_uni_w<W>_b<B>.json written by bench_transfer_parallel.py: one
adapter (hand_rorschach) transferred zero-shot onto the unicorn mesh at each of the
seven windows in the window-ablation table, with everything else held fixed.

WHAT IS BEING TESTED
    timing_table.tex asserts "the temporal component is parameter-free and adds no
    measurable cost ... since the blend is applied to cached conditioning tokens
    before the flow runs". That is reasoning, not measurement. If it holds, W=15
    costs what W=1 costs and the spread sits inside run-to-run noise.

FIELD NAMES ARE READ, NOT GUESSED
    The writer emits `inference_s`, `raster_s`, `peak_gpu_gib`. An earlier version of
    this file guessed at `infer_s`/`total_s` and would have reported every window as
    missing while looking like it had run fine. The keys below were copied off an
    actual output file.
"""
import json
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
WINDOWS = [1, 3, 5, 7, 11, 13, 15]
BATCHES = [1, 8]
NOISE = 0.05        # 5% of the mean; below this the window cost is not resolvable


def load(w, b):
    p = E / f'out/timing_uni_w{w}_b{b}.json'
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:                                            # noqa: BLE001
        return None
    return d if isinstance(d.get('inference_s'), (int, float)) else None


def main():
    data = {w: {b: load(w, b) for b in BATCHES} for w in WINDOWS}
    have = sum(1 for w in WINDOWS for b in BATCHES if data[w][b])
    print('Inference time vs MCFM window')
    print('hand_rorschach adapter, transferred zero-shot to the unicorn mesh')
    print(f'150 frames, 4 views, res 518.  {have}/{len(WINDOWS)*len(BATCHES)} cells reported\n')

    print(f'{"W":>3s}{"mode":>6s} | {"b=1 total":>10s}{"/frame":>9s} | '
          f'{"b=8 total":>10s}{"/frame":>9s} | {"speedup":>8s}{"peak GiB":>10s}')
    print('-' * 74)
    for w in WINDOWS:
        d1, d8 = data[w][1], data[w][8]
        mode = (d1 or d8 or {}).get('mcfm') or 'none'
        f = lambda d, k: f'{d[k]:.1f}s' if d else '--'
        g = lambda d, k: f'{d[k]:.2f}s' if d else '--'
        sp = f'{d1["inference_s"]/d8["inference_s"]:.2f}x' if (d1 and d8) else '--'
        pk = f'{d8["peak_gpu_gib"]:.1f}' if d8 else (f'{d1["peak_gpu_gib"]:.1f}' if d1 else '--')
        print(f'{w:>3d}{mode:>6s} | {f(d1,"inference_s"):>10s}'
              f'{g(d1,"inference_s_per_frame"):>9s} | {f(d8,"inference_s"):>10s}'
              f'{g(d8,"inference_s_per_frame"):>9s} | {sp:>8s}{pk:>10s}')

    for b in BATCHES:
        vals = [(w, data[w][b]['inference_s']) for w in WINDOWS if data[w][b]]
        if len(vals) < 3:
            continue
        # W=1 is a DIFFERENT checkpoint, trained without MCFM, so it carries
        # checkpoint and node variance on top of any window effect. The window
        # question is answered across the MCFM windows alone; W=1 is reported
        # beside them as the no-blend reference, not folded into the spread.
        mc = [(w, v) for w, v in vals if w != 1]
        if len(mc) < 2:
            continue
        xs = [v for _, v in mc]
        lo, hi, mu = min(xs), max(xs), sum(xs) / len(xs)
        spread = (hi - lo) / mu
        print(f'\nbatch={b}: MCFM windows W={[w for w, _ in mc]}')
        print(f'  min {lo:.1f}s  max {hi:.1f}s  mean {mu:.1f}s  spread {100*spread:.2f}%')
        print(f'  W={mc[0][0]} -> W={mc[-1][0]}: {mc[-1][1]-mc[0][1]:+.1f}s '
              f'({100*(mc[-1][1]-mc[0][1])/mc[0][1]:+.2f}%)')
        base = dict(vals).get(1)
        if base:
            print(f'  no-blend W=1 reference: {base:.1f}s '
                  f'({100*(mu-base)/base:+.1f}% vs the MCFM mean)')
        print('  -> ' + ('NO measurable window cost; timing_table.tex stands'
                         if spread < NOISE else
                         'window cost IS measurable; timing_table.tex needs correcting'))


if __name__ == '__main__':
    main()
