"""supervise_kl_grid.py <object> — keep every KL cell alive until it has a final.

THE RISK THIS COVERS. A grid cell trains 30 epochs against a --time=04:00:00 wall.
SLURM does NOT auto-requeue a job that runs out of clock: --requeue covers node
failure and preemption, not TIMEOUT. So a cell that dies at epoch 27 simply stops
and leaves a permanent hole in the grid -- which is exactly what happened to
c50_s25 on teapot_ceramic_crack_correct, and was only recovered because a
supervisor noticed. Node failure and OOM leave the same hole.

WHY A PLAIN RESUBMIT IS ENOUGH. rung30_kl_both.py checkpoints every epoch and has
GATE-resume, so resubmitting the same sbatch continues from the last completed
epoch rather than restarting. Recovery is cheap; the only hard part is noticing.

IDENTIFYING A CELL. All 17 jobs share --job-name=r30g_<obj> and write to
out/r30g_<obj>_<jobid>.log, so the FILENAME does not say which cell it is. Every
log's header line does:
    ### obj=... --w-kl <BC> --w-kl-self <BS>
That (BC, BS) pair is unique per cell and stable across resubmits, so it is the key.

DONE means the '[FINAL] rung30 PSNR ... SSIM ...' line. Epoch 30 alone is not
enough: the 150-frame evaluation runs after it and can still fail.
"""
import json, re, subprocess, sys, time
from pathlib import Path

OBJ = sys.argv[1]
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OUT, JOBS = E / 'out', E / 'jobs'
FR = [15, 25, 50, 75]
MAX_RETRY = 4
POLL = 180
DEADLINE = time.time() + 10 * 3600
LOG = OUT / f'kl_supervisor_{OBJ}.log'


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {OBJ}: {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


# ── wait for the probe, then read the ratios the grid was built from ─────────
RC = RS = None
while time.time() < DEADLINE and RC is None:
    ls = sorted(OUT.glob(f'r30probe_{OBJ}_*.log'))
    for p in ls:
        m = re.findall(r'loss/KLc=([0-9.e+-]+) +loss/KLs=([0-9.e+-]+)', p.read_text(errors='replace'))
        if m and 'EPOCH 6/6' in p.read_text(errors='replace'):
            RC, RS = float(m[-1][0]), float(m[-1][1])
    if RC is None:
        time.sleep(POLL)
if RC is None:
    log('probe never completed — cannot derive cell keys, exiting'); sys.exit(1)
log(f'probe ratios loss/KLc={RC} loss/KLs={RS}')


def beta(f, r):
    return float(f'{f/100/(1-f/100)*r:.6g}')


cells = {}
for fc in FR:
    for fs in FR:
        cells[(fc, fs)] = {'flag': f'--w-kl {beta(fc, RC):g} --w-kl-self {beta(fs, RS):g}',
                           'sbatch': JOBS / f'r30grid_{OBJ}_c{fc}_s{fs}.sbatch',
                           'retries': 0, 'final': None}
cells[(0, 0)] = {'flag': '--w-kl 0 --w-kl-self 0',
                 'sbatch': JOBS / f'r30grid_{OBJ}_c0_s0.sbatch',
                 'retries': 0, 'final': None}
log(f'tracking {len(cells)} cells (16 grid + 1 control)')

FINAL = re.compile(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)')
HDR = re.compile(r'--w-kl [0-9.eE+-]+ --w-kl-self [0-9.eE+-]+')


def scan():
    for f in OUT.glob(f'r30g_{OBJ}_*.log'):
        t = f.read_text(errors='replace')
        h, m = HDR.search(t), FINAL.findall(t)
        if not h:
            continue
        for k, c in cells.items():
            if c['flag'] == h.group(0) and m:
                c['final'] = (float(m[-1][0]), float(m[-1][1]))


def live():
    ids = sh(f'squeue -u rajhansini -h -n r30g_{OBJ} -o "%i"').split()
    out, unknown = set(), False
    for j in ids:
        p = OUT / f'r30g_{OBJ}_{j}.log'
        if p.exists():
            h = HDR.search(p.read_text(errors='replace'))
            if h:
                out.add(h.group(0)); continue
        unknown = True          # queued but no log yet — do not judge it absent
    return out, unknown, len(ids)


while time.time() < DEADLINE:
    scan()
    lv, unknown, nq = live()
    done = [k for k, c in cells.items() if c['final']]
    if len(done) == len(cells):
        log(f'ALL {len(cells)} CELLS HAVE FINALS'); break
    if not unknown:
        for k, c in cells.items():
            if c['final'] or c['flag'] in lv or c['retries'] >= MAX_RETRY:
                continue
            if not c['sbatch'].exists():
                continue
            c['retries'] += 1
            j = sh(f'sbatch --parsable {c["sbatch"]}')
            log(f'RESUBMIT c{k[0]}_s{k[1]} attempt {c["retries"]} -> job {j} '
                f'(no final, no live job — timeout or failure; resumes from checkpoint)')
            subprocess.run(['bash', '/net/projects/ranalab/rajhansini/joblog.sh', 'add', j,
                            f'r30g_{OBJ}_c{k[0]}_s{k[1]}_retry{c["retries"]}',
                            f'AUTO-RESUBMIT by supervisor. {c["flag"]} | PASS: [FINAL] line'],
                           capture_output=True)
    (OUT / f'kl_supervisor_{OBJ}.json').write_text(json.dumps(
        {f'c{k[0]}_s{k[1]}': {'final': v['final'], 'retries': v['retries']}
         for k, v in cells.items()}, indent=1))
    log(f'  {len(done)}/{len(cells)} final · {nq} in queue')
    time.sleep(POLL)

scan()
done = [k for k, c in cells.items() if c['final']]
log(f'=== end — {len(done)}/{len(cells)} cells have finals ===')
for k, c in sorted(cells.items()):
    if not c['final']:
        log(f'  MISSING c{k[0]}_s{k[1]} after {c["retries"]} retries')
