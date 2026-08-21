"""supervise_rung32.py — keep the 4 rung32 (context-window=3) runs alive to a final.

THE RISK. Every partition on this cluster caps at --time=04:00:00 and SLURM does
NOT auto-requeue a job that runs out of clock (--requeue covers node failure and
preemption, not TIMEOUT). Plain rung27 already takes ~1h54m at 150 frames; rung32
feeds cross-attention 3x the tokens (1029 -> 3087), so hitting the wall mid-run is
the expected case here, not the unlucky one.

WHY A PLAIN RESUBMIT IS ENOUGH. rung32_wide_context_lora.py checkpoints every epoch
and has GATE-resume (it refuses a checkpoint written against a different mesh or
target set), so resubmitting the same sbatch with the same --export continues from
the last completed epoch. Recovery is cheap; noticing is the hard part.

IDENTIFYING A RUN. One job per object, --job-name=r32_<obj>, so the object IS the
key. The run directory is rung27_l1_lp_cw3_* (the rung field stays 27; the _cw3 tag
carries the context window), matched back to its object via config.json's mesh.

DONE means final_eval.json exists. Epoch 30 alone is not enough -- the full-length
evaluation runs after it and can still fail.
"""
import json, subprocess, sys, time
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
OUT, JOBS, RUNS = E / 'out', E / 'jobs', E / 'runs'
SB = JOBS / 'rung32_w3_hero.sbatch'
MAX_RETRY, POLL = 6, 180
DEADLINE = time.time() + 30 * 3600
LOG = OUT / 'rung32_supervisor.log'

# mesh basename, targets dir suffix, n_frames -- spot_lava's mesh is spot_render_frame
# (not spot_lava_*) and pumpkin_rot carries _guan on both mesh and targets.
SPEC = {
    'spot_lava':      ('spot_render_frame',            'spot_lava',      150),
    'skull_lava':     ('skull_lava_render_frame',      'skull_lava',     150),
    'hand_rorschach': ('hand_rorschach_render_frame',  'hand_rorschach', 150),
    'pumpkin_rot':    ('pumpkin_rot_render_frame_guan','pumpkin_rot_guan',121),
}


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def done(obj):
    """A cw3 run for this object that reached final_eval.json."""
    for d in RUNS.glob('rung27_l1_lp_cw3_*'):
        cf, fe = d / 'config.json', d / 'final_eval.json'
        if not (cf.exists() and fe.exists()):
            continue
        try:
            c = json.load(open(cf))
        except Exception:
            continue
        if Path(c.get('mesh', '')).parent.parent.name == obj:
            return d
    return None


def submit(obj):
    m, tg, nf = SPEC[obj]
    env = (f'ALL,OBJ={obj},NFR={nf},'
           f'MESH={T2}/data/{obj}/mesh/{m}.obj,'
           f'GT={T2}/data/{obj}/frames_from_video,'
           f'TG={OUT}/gt_targets_{tg}/frames')
    j = sh(f'sbatch --parsable --job-name=r32_{obj} --export={env} {SB}')
    return j if j.isdigit() else None


live = dict(l.split('\t') for l in open('/tmp/r32_ids.txt').read().split('\n') if '\t' in l)
retry = {o: 0 for o in SPEC}
log(f'watching {live}')

while time.time() < DEADLINE:
    pending = [o for o in SPEC if not done(o)]
    if not pending:
        log('ALL 4 DONE')
        break
    q = sh('squeue -u rajhansini -h -o "%i"').split()
    for o in pending:
        jid = live.get(o)
        if jid and jid in q:
            continue                      # still running or queued
        st = sh(f'sacct -j {jid} -X -n -o State 2>/dev/null').split('\n')[0].strip() if jid else 'NONE'
        if retry[o] >= MAX_RETRY:
            log(f'{o}: GIVING UP after {MAX_RETRY} retries (last state {st})')
            continue
        retry[o] += 1
        nj = submit(o)
        if nj:
            live[o] = nj
            log(f'{o}: {st} with no final -> resubmitted as {nj} (retry {retry[o]}/{MAX_RETRY})')
        else:
            log(f'{o}: resubmit FAILED (state was {st})')
    time.sleep(POLL)
else:
    log('deadline reached')
