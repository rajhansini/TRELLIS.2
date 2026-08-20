"""
status_watch.py — one file to check on waking up, and a queue nurse.

TWO JOBS, both of which have to survive being left alone for hours.

1  WRITES out/STATUS.txt every cycle, overwritten in place, so there is exactly
   one file to `cat` and it is never stale by more than the cycle time. It reports
   what FINISHED, not what was submitted -- a submitted job that died is worse
   than useless as a status line, which is how 60 renders looked healthy right up
   until they were all dead.

2  RELEASES held jobs as capacity frees. 60 view renders are held so that the two
   deadline items (Guan's 1024 stills, and the skull_lava MCFM training that has
   to land before its own renders can run) are not stuck behind them. Holding them
   forever would just move the problem, so this releases a few at a time whenever
   the running count drops below --keep-running, which also stops the queue flood
   that put fairshare at 0.087 in the first place.

Everything it does is idempotent and read-only apart from `scontrol release`,
so it is safe to kill and restart.
"""
import argparse, json, re, subprocess, time
from datetime import datetime
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = E.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(E / 'out/STATUS.txt'))
ap.add_argument('--every', type=int, default=180, help='seconds between cycles')
ap.add_argument('--keep-running', type=int, default=8,
                help='release held jobs while fewer than this many of mine run')
ap.add_argument('--release-batch', type=int, default=4)
ap.add_argument('--cycles', type=int, default=400)
A = ap.parse_args()

OBJS = ['ancient_lady', 'plane_waves', 'teapot_crack', 'hand_rorschach', 'skull_lava',
        # batch 2: mesh chosen by silhouette IoU against the Kling still, every
        # match >= 0.988 with the runner-up 0.2-0.5 behind. monster == armadillo.
        'monster_lava_2', 'monster_rainbow', 'ancient_statue_clay', 'eagle_blackness',
        # batch 3, matched the same way: mushroom 97.9% (next 50.3), alien 98.8% (next 47.1)
        'mushroom_glow', 'alien_glow']
VIEWS = ['train', 'orbit360', 'orbit720', 'diagA', 'diagB', 'diagC']
NFR = {'ancient_lady': 135}
TRAIN = {  # object -> (r27 jobid, mcfm jobid) as first submitted
    'ancient_lady': ('2185421', '2185422'), 'plane_waves': ('2185423', '2185424'),
    'teapot_crack': ('2185425', '2185426'), 'hand_rorschach': ('2185427', '2185428'),
    'skull_lava': ('2185429', '2185430')}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def psnr_of(obj, arm):
    """Read the FINAL line out of whichever log actually produced it — a retried
    job has a new id, so globbing the run dir beats trusting a remembered jobid."""
    pre = 'r27hg' if arm == 'r27' else 'r27hm'
    best = None
    for p in sorted((E / 'out').glob(f'{pre}_*.log')):
        t = p.read_text(errors='replace')
        if f'OBJ={obj} ' not in t:
            continue
        m = re.findall(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)', t)
        if m:
            best = (float(m[-1][0]), float(m[-1][1]))
    return best


def main():
    for _ in range(A.cycles):
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        L = [f'STATUS  —  {now}   (refreshes every {A.every//60} min; this file is overwritten)',
             '=' * 88, '']

        # ---------------- queue ----------------
        run = sh("squeue -u rajhansini -h -t R -o '%j' | grep -vc nlp_ || true") or '0'
        pend = sh("squeue -u rajhansini -h -t PD -o '%j %r'")
        held = len([l for l in pend.splitlines() if 'JobHeldUser' in l])
        waiting = len([l for l in pend.splitlines() if 'JobHeldUser' not in l and 'nlp_' not in l])
        L += [f'QUEUE   running {run}    waiting {waiting}    held {held}', '']

        # ---------------- training ----------------
        L += ['TRAINING  (rung27 = cross+self-attn LoRA;  +MCFM = same, --mcfm v2_D)',
              '-' * 88,
              f'{"object":<18}{"r27 PSNR":>11}{"+MCFM PSNR":>13}{"delta":>9}   state']
        for o in OBJS:
            a, b = psnr_of(o, 'r27'), psnr_of(o, 'mcfm')
            if a and b:
                L.append(f'{o:<18}{a[0]:>11.3f}{b[0]:>13.3f}{b[0]-a[0]:>+9.3f}   both done')
            elif a:
                st = sh(f"squeue -u rajhansini -h -o '%t' -n mcf_{o}") or 'not queued'
                L.append(f'{o:<18}{a[0]:>11.3f}{"--":>13}{"--":>9}   MCFM {st}')
            else:
                L.append(f'{o:<18}{"--":>11}{"--":>13}{"--":>9}   running/queued')
        L.append('')

        # ---------------- renders ----------------
        L += ['RENDERS   frames on disk per (object, arm, view);  need N frames each',
              '-' * 88, f'{"object":<18}' + ''.join(f'{v:>11}' for v in VIEWS)]
        done_pairs = []
        for o in OBJS:
            n = NFR.get(o, 150)
            row = f'{o:<18}'
            for v in VIEWS:
                c = []
                for arm in ('r27', 'mcfm'):
                    d = E / 'out' / f'view_{o}_{arm}_{v}' / 'frames'
                    c.append(len(list(d.glob('*.png'))) if d.is_dir() else 0)
                ok = all(x == n for x in c)
                if ok:
                    done_pairs.append((o, v, n))
                row += f'{("BOTH" if ok else f"{c[0]}/{c[1]}"):>11}'
            L.append(row)
        L.append('')

        # ---------------- guan bundle ----------------
        L += ['GUAN BUNDLE  /net/projects/ranalab/rajhansini/TRELLIS.2/handoff/spot_lava_r27_mcfm',
              '-' * 88]
        for v in VIEWS:
            d = E / 'out' / f'guan_spot_lava_mcfm_{v}' / 'frames'
            c = len(list(d.glob('*.png'))) if d.is_dir() else 0
            L.append(f'  1024px {v:<10} {c:>4}/150 frames' + ('   DONE' if c == 150 else ''))
        L.append('')

        # ---------------- panel videos ----------------
        vids = sorted((E / 'out/VIEWS').glob('*.mp4')) if (E / 'out/VIEWS').is_dir() else []
        L += [f'PANEL VIDEOS  out/VIEWS/   {len(vids)} of 30 built',
              '-' * 88]
        for p in vids:
            L.append(f'  {p.name}  {p.stat().st_size/1e6:.1f} MB')
        if not vids:
            L.append('  (none yet — built once both arms of a view have all frames)')
        L += ['', '=' * 88,
              'next: 30 panel videos (GT | frozen | rung27 | rung27+MCFM), then the artifact.',
              f'raw logs: {E}/out/rview_*.log , r27hg_*.log , r27hm_*.log']

        Path(A.out).write_text('\n'.join(L) + '\n')

        # ---------------- nurse the queue ----------------
        try:
            nrun = int(sh("squeue -u rajhansini -h -t R -o '%j' | grep -vc nlp_ || echo 0") or 0)
            if nrun < A.keep_running:
                ids = [l.split()[0] for l in
                       sh("squeue -u rajhansini -h -t PD -o '%i %r'").splitlines()
                       if 'JobHeldUser' in l][:A.release_batch]
                if ids:
                    subprocess.run(['scontrol', 'release'] + ids, capture_output=True)
        except Exception:
            pass

        time.sleep(A.every)


if __name__ == '__main__':
    main()
