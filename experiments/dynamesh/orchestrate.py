"""
orchestrate.py — drive every object from trained checkpoint to published-ready page,
without a human in the loop.

WHY THIS EXISTS
  Each stage was already automatic in isolation: training resumes, renders and panels
  requeue, the watchdog resubmits infrastructure failures. What was NOT automatic was
  the hand-off BETWEEN stages — a run directory is a config hash, so render jobs could
  not be queued until training had started and written it. That gap was filled by me
  noticing, which is exactly the thing that cannot happen overnight.

  Every cycle this asks four questions and acts on them:
    1. trained but no renders queued   -> submit 12 render jobs, gated on nothing
                                          (training is already done, so no dependency)
    2. rendered but no panel videos    -> submit the panel job
    3. panels built but not web-encoded-> encode them
    4. anything new encoded            -> rebuild the artifact HTML

  It is idempotent at every step: it looks at what is ON DISK, never at what it
  remembers submitting, so restarting it is free and double-submission is impossible.

WHAT IT DELIBERATELY DOES NOT DO
  Publish. Publishing is an outward-facing action, and a loop that republishes on its
  own could push a half-finished page over a good one. It writes the HTML and records
  readiness; a scheduled session publishes it.
"""
import json, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
LOG = E / 'out' / 'ORCHESTRATOR.log'
VIEWS = [('train', 0, 0, 0), ('orbit360', 0, 15, 1), ('orbit720', 0, 15, 2),
         ('diagA', 45, 25, 0), ('diagB', 135, -20, 0), ('diagC', 225, 30, 0)]
NFR = {'ancient_lady': 135}

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--every', type=int, default=300)
ap.add_argument('--minutes', type=float, default=225)
ap.add_argument('--max-queued', type=int, default=40,
                help='do not add render jobs beyond this many of mine pending')
A = ap.parse_args()


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout


def note(m):
    line = f'{datetime.now():%m-%d %H:%M}  {m}'
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def objects():
    """Objects built by THIS pipeline, and only those.

    build_summary.json is written solely by build_targets_hero.py, so keying on it
    excludes the 29 older target directories that share the gt_targets_* prefix --
    dead experiments (spot_REAR_BAD, teapot_960_BAD_normalised_mesh) and the earlier
    Guan-rotation set, none of which this orchestrator should ever act on.
    """
    return sorted(p.name.replace('gt_targets_', '')
                  for p in (E / 'out').glob('gt_targets_*')
                  if (p / 'gt_targets.json').exists() and (p / 'build_summary.json').exists())


def trained(o):
    """(r27_run, mcfm_run) once BOTH arms have a finished checkpoint, else None."""
    out = {}
    for pre, arm in (('r27hg', 'r27'), ('r27hm', 'mcfm')):
        for p in sorted((E / 'out').glob(f'{pre}_*.log')):
            t = p.read_text(errors='replace')
            if f'OBJ={o} ' not in t or '[FINAL] rung' not in t:
                continue
            m = re.findall(r'runs/rung27_[A-Za-z0-9_.+-]+', t)
            if m and (E / m[-1] / 'ckpts' / 'lora_best.pt').exists():
                out[arm] = m[-1]
    return (out['r27'], out['mcfm']) if len(out) == 2 else None


def n_frames(o, arm, v):
    d = E / 'out' / f'view_{o}_{arm}_{v}' / 'frames'
    return len(list(d.glob('*.png'))) if d.is_dir() else 0


def queued_tags():
    return set(re.findall(r'TAG=(\S+)', sh("squeue -u rajhansini -h -o '%o'"))) | \
           {l.strip() for l in sh("squeue -u rajhansini -h -o '%j'").splitlines()}


def cycle():
    qn = len([l for l in sh("squeue -u rajhansini -h -t PD -o '%j'").splitlines()
              if not l.startswith('nlp_')])
    running_names = sh("squeue -u rajhansini -h -o '%j'").split()
    new_html = False

    for o in objects():
        n = NFR.get(o, 150)
        runs = trained(o)
        if not runs:
            continue

        # ---- 1. renders missing?
        need = [(arm, v) for arm in ('r27', 'mcfm') for v, *_ in VIEWS
                if n_frames(o, arm, v) < n]
        if need and qn < A.max_queued:
            # Dedup on the FULL object name in the TAG, not a 6-char prefix:
            # o[:6] made 'ancient_lady_teaser' and 'ancient_lady_effect_1' collide
            # as 'ancien', so one object's in-flight jobs suppressed another's and
            # the same 12 renders were queued twice.
            inflight = f'view_{o}_' in sh("squeue -u rajhansini -h -o '%o'")
            if not inflight:
                for arm, v in need:
                    y, e, t = next((a, b, c) for vv, a, b, c in VIEWS if vv == v)
                    rd = runs[0] if arm == 'r27' else runs[1]
                    sh(f'cd {E} && sbatch --parsable --job-name=v_{arm}_{v}_{o[:6]} '
                       f'--export=ALL,RUN={E}/{rd},TAG=view_{o}_{arm}_{v},NFR={n},'
                       f'YAW0={y},ELEV={e},TURNS={t} jobs/render_view.sbatch')
                    qn += 1
                note(f'{o}: queued {len(need)} render jobs')
            continue

        if need:
            continue                                   # still rendering

        # ---- 2. panels missing?
        vids = [E / 'out/VIEWS' / f'{o}_{v}_GT_frozen_r27_mcfm.mp4' for v, *_ in VIEWS]
        short = [p for p in vids if not p.exists() or p.stat().st_size < 100_000]
        if short:
            if not any(nm == f'pan_{o[:8]}' for nm in running_names):
                sh(f'cd {E} && sbatch --parsable --job-name=pan_{o[:8]} '
                   f'--export=ALL,OBJ={o} jobs/panels_one.sbatch')
                note(f'{o}: queued panel job ({len(short)}/6 missing)')
            continue

        # ---- 3. web encode
        for p in vids:
            w = E / 'out/VIEWS_WEB' / p.name
            if not w.exists():
                (E / 'out/VIEWS_WEB').mkdir(exist_ok=True)
                r = subprocess.run(
                    ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(p),
                     '-vf', 'scale=1120:-2', '-c:v', 'libx264', '-crf', '31',
                     '-preset', 'slow', '-pix_fmt', 'yuv420p',
                     '-movflags', '+faststart', str(w)], capture_output=True)
                if r.returncode == 0:
                    new_html = True
        if new_html:
            note(f'{o}: encoded for web')

    # ---- 4. refresh PSNR + rebuild the page
    if new_html:
        res = {}
        for pre, arm in (('r27hg', 'r27'), ('r27hm', 'mcfm')):
            for p in sorted((E / 'out').glob(f'{pre}_*.log')):
                t = p.read_text(errors='replace')
                m = re.search(r'OBJ=(\S+) ', t)
                f = re.findall(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)', t)
                if m and f:
                    res.setdefault(m.group(1), {})[arm] = [float(f[-1][0]), float(f[-1][1])]
        json.dump(res, open(E / 'out/psnr_all.json', 'w'), indent=1)
        out = sh(f'cd {E} && /net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python '
                 f'jobs/build_views_artifact.py {E}/out/six_cameras.html out/VIEWS_WEB')
        note('rebuilt page: ' + (out.strip().splitlines() or ['(no output)'])[0])
        (E / 'out' / 'ARTIFACT_READY').write_text(
            f'{datetime.now():%Y-%m-%d %H:%M}\n{len(list((E/"out/VIEWS_WEB").glob("*.mp4")))} videos\n')


def main():
    stop = time.time() + A.minutes * 60
    note(f'orchestrator up until {datetime.fromtimestamp(stop):%H:%M}')
    while time.time() < stop:
        try:
            cycle()
        except Exception as ex:
            note(f'cycle error: {type(ex).__name__}: {ex}')
        time.sleep(A.every)
    note('orchestrator exiting (successor should take over)')


if __name__ == '__main__':
    main()
