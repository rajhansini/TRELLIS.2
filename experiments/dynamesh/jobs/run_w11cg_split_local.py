#!/usr/bin/env python3
"""run_w11cg_split_local.py -- run this host's half of the W=11 copy-GT cells.

The object list is fixed in out/logs/local/split_local.txt and the slurm half was
cancelled for exactly those objects, so the two paths never write the same file.
Cells whose json already exists are skipped, so this is safe to re-run.
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
PY = '/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python'
LOGS = f'{E}/out/logs/local'
D, ARM = 'FULLRATE_W11CG', '27g'

objs = [l.strip() for l in open(f'{LOGS}/split_local.txt') if l.strip()]
todo = [o for o in objs if not (os.path.exists(f'{E}/out/{D}/{o}.json')
                                and os.path.getsize(f'{E}/out/{D}/{o}.json') > 0)]
print(f'{len(todo)} of {len(objs)} local cells to run', flush=True)

t0 = time.time()
done = [0]

def run(obj):
    out = f'{E}/out/{D}/{obj}.json'
    with open(f'{LOGS}/{D}.{obj}.log', 'w') as fh:
        r = subprocess.run([PY, '-u', f'{E}/jobs/fullrate_metrics_arm.py',
                            '--obj', obj, '--arm', ARM, '--out', out],
                           cwd=E, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0
    done[0] += 1
    el = time.time() - t0
    print(f'[{done[0]:2d}/{len(todo)}] {"OK  " if ok else "FAIL"} {obj}'
          f'   {el/60:.1f}m elapsed, ~{el/done[0]*(len(todo)-done[0])/60:.1f}m left', flush=True)
    return (ok, obj)

with ThreadPoolExecutor(max_workers=6) as ex:
    res = list(ex.map(run, todo))

bad = [o for ok, o in res if not ok]
print(f'\nlocal half done in {(time.time()-t0)/60:.1f} min; {len(res)-len(bad)} ok, {len(bad)} failed', flush=True)
for b in bad:
    print('  FAIL ' + b, flush=True)
sys.exit(1 if bad else 0)
