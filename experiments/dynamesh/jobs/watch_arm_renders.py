#!/usr/bin/env python3
"""watch_arm_renders.py -- failure monitor + self-heal for the r19/r37/r31 render fleet.

WHY THIS EXISTS, AND WHY THE FIRST MONITOR DID NOT WORK
  The first watch used `sacct -o JobName%40,State | grep -E '^(r19v_|...)'`. sacct
  RIGHT-PADS a %40 field, so every line begins with spaces and the anchored grep
  matched nothing, ever. It ran clean through the whole fleet and would have stayed
  silent through a total wipeout -- the exact trap where silence is indistinguishable
  from success. `-P` (pipe-delimited, unpadded) is used here instead, and the filter
  is asserted against a known-live job name at startup rather than trusted.

TWO FAILURE MODES, NOT ONE
  1. Terminal-bad state (FAILED / TIMEOUT / OUT_OF_MEMORY / NODE_FAIL).
  2. COMPLETED with too few frames. render_arm.sbatch exits 0 on its idempotent
     "ALREADY COMPLETE" path, so a job that renders nothing at all can still report
     COMPLETED. Frames on disk are the pass condition, never the exit status.

SELF-HEAL. Either mode triggers `submit_arm_renders.sh ARM=.. ONLY=..`, which is
idempotent: it re-submits only cells whose frames are short AND whose job name is not
already queued. Retries are capped per CELL at MAXTRY so a genuinely broken cell
cannot loop forever, and the cap is reported rather than swallowed.
"""
import json, os, subprocess, sys, time
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
SINCE  = os.environ.get('SINCE', '2026-08-29')
EVERY  = int(os.environ.get('EVERY', '300'))
MAXTRY = int(os.environ.get('MAXTRY', '3'))
RETRY_F = E / 'out' / 'arm_render_retries.json'
PREFIX = ('r19v_', 'r37v_', 'r31v_', 'r31mv_')
TAG = {'r19v': '19', 'r37v': '37', 'r31v': '31', 'r31mv': '31m'}
BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL')

NFR = {}
for line in (E / 'jobs' / 'rung37_objects.tsv').read_text().splitlines()[1:]:
    if line.strip():
        p = line.split('\t')
        NFR[p[1]] = int(p[2])


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def split_name(jn):
    """r31mv_tie_fighter_bw_diagB -> ('r31mv', 'tie_fighter_bw', 'diagB')"""
    arm, rest = jn.split('_', 1)
    obj, view = rest.rsplit('_', 1)
    return arm, obj, view


def frames(arm, obj, view):
    d = E / 'out' / f'view_{obj}_{TAG[arm]}_{view}' / 'frames'
    return len(list(d.glob('*.png'))) if d.is_dir() else 0


VIEWS = ('train', 'diagA', 'diagB', 'diagC')


def done_cells(arm):
    """Complete cells at the FOUR fixed cameras only.

    `glob(f'view_*_{tag}_*')` is wrong here and was: it also matches the u0..u6
    unseen-view dirs from earlier batches, which made r31 report 215/96 and fire
    ARM COMPLETE while 27 of its jobs were still queued. A progress line that can
    read 100% while work is outstanding is worse than no progress line.
    """
    n = 0
    for obj, want in NFR.items():
        for v in VIEWS:
            d = E / 'out' / f'view_{obj}_{TAG[arm]}_{v}' / 'frames'
            if d.is_dir() and len(list(d.glob('*.png'))) >= want:
                n += 1
    return n


def sacct_rows():
    out = sh('sacct', '-S', SINCE, '-n', '-X', '-P', '-o', 'JobID,JobName,State')
    rows = []
    for line in out.splitlines():
        f = line.split('|')
        if len(f) >= 3 and f[1].startswith(PREFIX):
            rows.append((f[0], f[1], f[2].split()[0]))
    return rows


def queued():
    return set(sh('squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%j').split())


retries = json.loads(RETRY_F.read_text()) if RETRY_F.exists() else {}
seen, capped, arm_done = set(), set(), set()

rows = sacct_rows()
if not rows:
    print('MONITOR ABORT: sacct matched 0 fleet jobs -- the filter is wrong, '
          'not the fleet. Refusing to run a watch that cannot see failures.')
    sys.exit(1)
print(f'monitor up: {len(rows)} fleet jobs visible to sacct, poll {EVERY}s, '
      f'retry cap {MAXTRY}/cell')

while True:
    rows = sacct_rows()
    live = queued()
    fix = {}                      # (arm, obj) -> reason, deduped per object
    latest = {}                   # keep only the newest row per job name
    for jid, jn, st in rows:
        latest[jn] = (jid, st)

    for jn, (jid, st) in sorted(latest.items()):
        if st in ('PENDING', 'RUNNING', 'REQUEUED', 'RESIZING', 'SUSPENDED'):
            continue
        arm, obj, view = split_name(jn)
        n, want = frames(arm, obj, view), NFR.get(obj, 150)
        if st in BAD:
            key = f'{jn}|{st}|{jid}'
            if key not in seen:
                seen.add(key)
                print(f'BAD   {jn}  {st}  ({n}/{want} frames)')
            if n < want:
                fix[(arm, obj)] = st
        elif st == 'COMPLETED' and n < want:
            # exit 0 with short output: the silent mode, worth its own label
            key = f'{jn}|SHORT|{n}'
            if key not in seen:
                seen.add(key)
                print(f'SHORT {jn}  COMPLETED but {n}/{want} frames on disk')
            fix[(arm, obj)] = 'SHORT'

    for (arm, obj), why in sorted(fix.items()):
        cell = f'{arm}/{obj}'
        if any(jn.startswith(f'{arm}_{obj}_') or split_name(jn)[:2] == (arm, obj)
               for jn in live):
            continue                                  # already back in the queue
        tries = retries.get(cell, 0)
        if tries >= MAXTRY:
            if cell not in capped:
                capped.add(cell)
                print(f'CAPPED {cell} after {tries} attempts ({why}) -- needs a human')
            continue
        retries[cell] = tries + 1
        RETRY_F.write_text(json.dumps(retries, indent=1))
        r = subprocess.run(['bash', str(E / 'jobs' / 'submit_arm_renders.sh')],
                           capture_output=True, text=True,
                           env={**os.environ, 'ARM': arm.rstrip('v') if arm != 'r31mv'
                                else 'r31m', 'ONLY': obj}, cwd=str(E))
        got = [l.strip() for l in r.stdout.splitlines() if l.startswith('  2')]
        print(f'RESUBMIT {cell} ({why}) attempt {tries + 1}/{MAXTRY}: '
              f'{len(got)} job(s){"" if got else " -- SUBMITTER RETURNED NOTHING"}')

    for arm in ('r19v', 'r37v', 'r31v', 'r31mv'):
        want = {'r19v': 168, 'r37v': 168, 'r31v': 96, 'r31mv': 96}[arm]
        have = done_cells(arm)
        if have >= want and arm not in arm_done:
            arm_done.add(arm)
            print(f'ARM COMPLETE {arm}: {have}/{want} view dirs')

    if not any(jn.startswith(PREFIX) for jn in live):
        tot = {a: done_cells(a) for a in TAG}
        print(f'FLEET DRAINED: queue empty. view dirs '
              f'r19={tot["r19v"]}/168 r37={tot["r37v"]}/168 '
              f'r31={tot["r31v"]}/96 r31m={tot["r31mv"]}/96')
        break
    time.sleep(EVERY)
