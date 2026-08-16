"""
supervise_ceramic_grid.py — keep all 16 KL cells alive until every one has a final.

WHY THIS IS NEEDED, CONCRETELY. Each cell needs 30 epochs at ~402 s, which is
3 h 21 of training plus ~10 min of model load and gates: about 3 h 31 against a
--time=04:00:00 wall. That is a 29-minute margin, and it evaporates the moment
cluster contention rises. SLURM does NOT auto-requeue a TIMEOUT job -- --requeue
covers node failure and preemption, not running out of clock -- so a cell that
hits the wall at epoch 27 simply stops and leaves a hole in the grid.

rung30_kl_both.py checkpoints every epoch and has GATE-resume, so a plain
resubmit of the same sbatch continues from the last completed epoch rather than
restarting. That makes recovery cheap: this script just has to notice and
resubmit.

IDENTIFYING A CELL. All 16 share --job-name=r30g_teapot_ceramic_crack_correct and
write to out/r30g_teapot_ceramic_crack_correct_<jobid>.log, so the filename does
NOT say which cell it is. Every log's header line does:
    ### obj=... --w-kl <BC> --w-kl-self <BS>
so cells are keyed by that (BC, BS) pair, which is unique per cell and stable
across resubmits.

WHAT COUNTS AS DONE. The '[FINAL] rung30 PSNR ... SSIM ...' line, in any log
belonging to that cell. Epoch 30 alone is not enough -- the 150-frame eval runs
after it and can still fail.
"""
import json, re, subprocess, sys, time
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OUT, JOBS = E / 'out', E / 'jobs'
OBJ = 'teapot_ceramic_crack_correct'
RC, RS = 5.426, 1.665            # probe 2182583, epoch 6
FRACS = [15, 25, 50, 75]
MAX_RETRY = 4                    # 4 x ~3.5 h covers any plausible wall/preempt chain
POLL = 180
DEADLINE = time.time() + 7 * 3600
STATE = OUT / 'ceramic_grid_supervisor.json'
LOG = OUT / 'ceramic_grid_supervisor.log'


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def beta(f, r):
    return float(f'{f/100/(1-f/100)*r:.6g}')


# cell key -> the exact strings that appear in that cell's log header and sbatch
cells = {}
for fc in FRACS:
    for fs in FRACS:
        cells[(fc, fs)] = {
            'flag': f'--w-kl {beta(fc, RC):g} --w-kl-self {beta(fs, RS):g}',
            'sbatch': JOBS / f'r30grid_{OBJ}_c{fc}_s{fs}.sbatch',
            'retries': 0, 'final': None,
        }

FINAL = re.compile(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)')
HDR = re.compile(r'--w-kl [0-9.eE+-]+ --w-kl-self [0-9.eE+-]+')


def scan():
    """Map every log on disk to its cell, and record any final it carries."""
    seen = {}
    for f in OUT.glob(f'r30g_{OBJ}_*.log'):
        try:
            t = f.read_text(errors='replace')
        except OSError:
            continue
        h = HDR.search(t)
        if not h:
            continue
        m = FINAL.findall(t)
        seen.setdefault(h.group(0), []).append(
            (f, float(m[-1][0]) if m else None, float(m[-1][1]) if m else None))
    for k, c in cells.items():
        for f, ps, ss in seen.get(c['flag'], []):
            if ps is not None:
                c['final'] = (ps, ss, f.name)
    return seen


def running_flags():
    """Which cells currently have a live job, by reading each running job's log."""
    ids = sh(f'squeue -u rajhansini -h -n r30g_{OBJ} -o "%i"').split()
    live = set()
    for j in ids:
        p = OUT / f'r30g_{OBJ}_{j}.log'
        if p.exists():
            h = HDR.search(p.read_text(errors='replace'))
            if h:
                live.add(h.group(0))
        else:
            live.add(f'PENDING:{j}')      # queued, log not written yet
    return live, len(ids)


log(f'=== supervisor start — {len(cells)} cells, deadline in 7h ===')
while time.time() < DEADLINE:
    scan()
    live, nq = running_flags()
    done = [k for k, c in cells.items() if c['final']]
    if len(done) == len(cells):
        log(f'ALL {len(cells)} CELLS HAVE FINALS')
        break

    # A cell needs resubmitting when it has no final AND no live job. The
    # PENDING: entries keep a just-queued job from being counted as absent
    # before its log exists, which would double-submit it.
    pending_unknown = any(f.startswith('PENDING:') for f in live)
    for k, c in cells.items():
        if c['final'] or c['flag'] in live:
            continue
        if pending_unknown:
            continue          # can't tell yet; wait for the log to appear
        if c['retries'] >= MAX_RETRY:
            continue
        c['retries'] += 1
        j = sh(f'sbatch --parsable {c["sbatch"]}')
        log(f'RESUBMIT c{k[0]}_s{k[1]} (attempt {c["retries"]}) -> job {j} '
            f'— resumes from its last checkpoint')
        subprocess.run(
            ['bash', '/net/projects/ranalab/rajhansini/joblog.sh', 'add', j,
             f'r30g_{OBJ}_c{k[0]}_s{k[1]}_retry{c["retries"]}',
             f'AUTO-RESUBMIT by supervisor: cell had no final and no live job '
             f'(wall-clock timeout or failure). Resumes from checkpoint. '
             f'{c["flag"]} | PASS: [FINAL] line in log'], capture_output=True)

    STATE.write_text(json.dumps(
        {f'c{k[0]}_s{k[1]}': {'final': v['final'], 'retries': v['retries']}
         for k, v in cells.items()}, indent=1))
    log(f'  {len(done)}/{len(cells)} final · {nq} in queue')
    time.sleep(POLL)

scan()
done = [k for k, c in cells.items() if c['final']]
log(f'=== supervisor end — {len(done)}/{len(cells)} cells have finals ===')
print('RESULTS_JSON_START')
print(json.dumps({f'c{k[0]}_s{k[1]}': {
    'beta_c': beta(k[0], RC), 'beta_s': beta(k[1], RS),
    'psnr': c['final'][0] if c['final'] else None,
    'ssim': c['final'][1] if c['final'] else None,
    'retries': c['retries']} for k, c in cells.items()}, indent=1))
print('RESULTS_JSON_END')
sys.exit(0 if len(done) == len(cells) else 1)
