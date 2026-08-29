#!/usr/bin/env python3
"""run_window_cells_local.py -- run the missing window metric cells on this host.

The slurm queue estimated 3 h before these would even start; the work itself is
~45 s of NFS PNG reads per cell and needs no GPU, so it runs here instead. Cells
are skipped if their json already exists, so this is safe to re-run.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
PY = '/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python'
LOGS = f'{E}/out/logs/local'
os.makedirs(LOGS, exist_ok=True)

OBJS = json.load(open(f'{E}/out/fullrate_table_batchALL.json'))['objects']
CELLS = [(a, d, o) for a, d in (('27f', 'FULLRATE_W7'), ('27g', 'FULLRATE_W11CG'))
         for o in OBJS]
todo = [c for c in CELLS if not (os.path.exists(f'{E}/out/{c[1]}/{c[2]}.json')
                                 and os.path.getsize(f'{E}/out/{c[1]}/{c[2]}.json') > 0)]
for _, d, _ in CELLS:
    os.makedirs(f'{E}/out/{d}', exist_ok=True)
print(f'{len(todo)} cells to run ({len(CELLS) - len(todo)} already done)', flush=True)

t0 = time.time()
done = [0]

def run(cell):
    arm, d, obj = cell
    out = f'{E}/out/{d}/{obj}.json'
    log = f'{LOGS}/{d}.{obj}.log'
    with open(log, 'w') as fh:
        r = subprocess.run([PY, '-u', f'{E}/jobs/fullrate_metrics_arm.py',
                            '--obj', obj, '--arm', arm, '--out', out],
                           cwd=E, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
    done[0] += 1
    el = time.time() - t0
    rate = el / done[0]
    print(f'[{done[0]:2d}/{len(todo)}] {"OK  " if ok else "FAIL"} {d} {obj}'
          f'   {el/60:.1f}m elapsed, ~{rate*(len(todo)-done[0])/60:.1f}m left', flush=True)
    return (ok, d, obj)

with ThreadPoolExecutor(max_workers=int(os.environ.get("NW","20"))) as ex:
    res = list(ex.map(run, todo))

bad = [f'{d}/{o}' for ok, d, o in res if not ok]
print(f'\ndone in {(time.time()-t0)/60:.1f} min; {len(res)-len(bad)} ok, {len(bad)} failed', flush=True)
for b in bad:
    print('  FAIL ' + b, flush=True)
for d in ('FULLRATE_W7', 'FULLRATE_W11CG'):
    n = len([f for f in os.listdir(f'{E}/out/{d}') if f.endswith('.json')])
    print(f'{d}: {n}/42', flush=True)
sys.exit(1 if bad else 0)
