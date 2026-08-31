#!/usr/bin/env python3
"""watch_window_hi.py -- failure monitor for the 84-job W=13/15 training fleet.

WHY THIS IS NOT THE RENDER MONITOR AGAIN. On a render fleet a TIMEOUT is a failure.
Here it is the NORMAL outcome: fig27.sbatch has a 4h wall because partition MaxTime
IS 4h, and 30 epochs takes 3-6h, so nearly every cell times out at least once and
resumes from ckpts on the watchdog's resubmit. Emitting one event per TIMEOUT would
be ~84 notifications that all mean "working as designed", and the real failures would
drown in them. So TIMEOUT is counted, never announced, and only its EXHAUSTION is.

WHAT IS WORTH WAKING SOMEONE FOR
  1. FAILED / OUT_OF_MEMORY / NODE_FAIL -- a real crash. W=15 stacks a wider window
     than anything trained before, so OOM is the specific risk this fleet carries.
  2. A GATE failure. watchdog.py deliberately REFUSES to resubmit these ("needs a
     fix"), so a gated cell is stuck forever and silent unless something says so.
  3. Retry exhaustion -- the watchdog gave up after 3 attempts.

PASS CONDITION is final_eval.json in the run directory, never job state: a cell that
TIMEOUTs at epoch 29 and is resubmitted is not done, and a cell whose last job shows
COMPLETED after a resume IS.
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
SINCE = os.environ.get('SINCE', '2026-08-29')
EVERY = int(os.environ.get('EVERY', '600'))
MODES = ('v2_H', 'v2_I')
BAD = ('FAILED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL')

OBJS = [l.split('\t')[1] for l in
        (E / 'jobs' / 'rung37_objects.tsv').read_text().splitlines()[1:] if l.strip()]
TOTAL = len(OBJS) * len(MODES)


def sh(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout


def done_cells():
    n = 0
    for m in MODES:
        for o in OBJS:
            for c in glob.glob(str(E / f'runs/rung27_l1_lp_mcfm{m}_all_qkvo+sa_r4_s42_*/config.json')):
                try:
                    d = json.load(open(c))
                except Exception:
                    continue
                if str(d.get('mcfm')) != m or d.get('epochs') != 30:
                    continue
                if f'/data/{o}/' not in (d.get('gt_dir') or ''):
                    continue
                if os.path.exists(os.path.join(os.path.dirname(c), 'final_eval.json')):
                    n += 1
                    break
    return n


def rows():
    out = sh('sacct', '-S', SINCE, '-n', '-X', '-P', '-o', 'JobID,JobName,State')
    r = []
    for line in out.splitlines():
        f = line.split('|')
        if len(f) >= 3 and f[1].startswith(('27_v2_H_', '27_v2_I_')):
            r.append((f[0], f[1], f[2].split()[0]))
    return r


if not rows():
    print('MONITOR ABORT: sacct matched 0 W=13/15 jobs -- filter is wrong, not the fleet.')
    sys.exit(1)
print(f'monitor up: W=13/15 fleet, {TOTAL} cells, {done_cells()} already complete, '
      f'poll {EVERY}s. TIMEOUT is expected and will NOT be reported.')

seen, milestone, timeouts = set(), 0, {}
while True:
    live = set(sh('squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%j').split())
    for jid, jn, st in rows():
        if st in ('PENDING', 'RUNNING', 'REQUEUED', 'SUSPENDED', 'COMPLETED'):
            continue
        if st == 'TIMEOUT':
            # Count UNIQUE JOB IDS, not sacct rows. The same TIMEOUT row is returned
            # by every poll, so incrementing per row made one timeout look like four
            # after four polls and fired a bogus "past the watchdog cap" alarm on two
            # cells that had in fact already resumed and finished. `seen` was applied
            # to BAD states but not to this branch.
            if f'to:{jid}' in seen:
                continue
            seen.add(f'to:{jid}')
            timeouts[jn] = timeouts.get(jn, 0) + 1
            if timeouts[jn] >= 4 and f'exh:{jn}' not in seen:
                seen.add(f'exh:{jn}')
                print(f'EXHAUSTED {jn}: {timeouts[jn]} timeouts, past the watchdog cap '
                      f'-- this cell will not finish on its own')
            continue
        if st in BAD and f'{jid}:{jn}' not in seen:
            seen.add(f'{jid}:{jn}')
            logs = sorted((E / 'out' / 'FIGRUNS').glob(f'{jn}_{jid}.log'))
            tail = ''
            if logs:
                txt = logs[-1].read_text(errors='replace')
                hits = [l for l in txt.splitlines()
                        if 'FAILED' in l or 'Error' in l or 'out of memory' in l]
                tail = ' | ' + hits[-1][:160] if hits else ''
            print(f'BAD {jn} {st}{tail}')
    # a gated cell is refused by the watchdog, so it is stuck and silent
    for f in (E / 'out' / 'FIGRUNS').glob('27_v2_[HI]_*.log'):
        if f.name in seen:
            continue
        try:
            t = f.read_text(errors='replace')
        except Exception:
            continue
        for line in t.splitlines():
            if 'GATE' in line and 'FAILED' in line:
                seen.add(f.name)
                print(f'GATE FAILURE {f.name}: {line.strip()[:180]} '
                      f'-- watchdog will NOT resubmit this')
                break
    d = done_cells()
    if d >= milestone + 10:
        milestone = d - (d % 10)
        print(f'PROGRESS {d}/{TOTAL} cells have final_eval.json '
              f'({len(timeouts)} cells have timed out at least once, which is normal)')
    if d >= TOTAL:
        print(f'W=13/15 COMPLETE: {d}/{TOTAL} cells trained')
        break
    if not any(n.startswith(('27_v2_H_', '27_v2_I_')) for n in live):
        print(f'QUEUE EMPTY with {d}/{TOTAL} complete -- {TOTAL - d} cells did not '
              f'finish and nothing is queued to retry them')
        break
    time.sleep(EVERY)
