"""
build_all_panels.py — build every (object, view) panel video whose renders are complete.

Skips pairs that are not finished rather than failing, so it can be re-run as more
renders land and it only does the new work. Skips videos that already exist and
already carry the right frame count, so a re-run is cheap.

Bounded concurrency: PIL compositing 150 frames x 4 panels is CPU-bound, and running
all of them at once on a login node is antisocial.
"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path(__file__).resolve().parent
PY = sys.executable

# ---------------------------------------------------------------- ffmpeg lookup
# THREE ffmpegs on this cluster and only two of them can do the job:
#   /usr/bin/ffmpeg                    libx264 YES  -- but LOGIN NODE ONLY
#   conda_envs/trellis2/bin/ffmpeg     libx264 NO   -- openh264 only, exits 8
#   conda_envs/richards_distillation/  libx264 YES  -- shared storage, all nodes
# Picking by PATH gets whichever comes first and fails on compute nodes with a
# bare "Encoder not found"; picking by first-existing gets the encoder-less one.
# So probe for the ENCODER, not the binary, and cache the answer.
_FFMPEG = None


def ffprobe_bin():
    """Sibling of whichever ffmpeg we settled on — same build, same node."""
    return str(Path(ffmpeg_bin()).with_name('ffprobe'))


def ffmpeg_bin():
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    import shutil as _sh
    cands = ['/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin/ffmpeg',
             '/usr/bin/ffmpeg', _sh.which('ffmpeg')]
    for c in cands:
        if not c or not Path(c).exists():
            continue
        try:
            enc = subprocess.run([c, '-hide_banner', '-encoders'],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        if 'libx264' in enc:
            _FFMPEG = c
            return c
    raise SystemExit('FAILED: no ffmpeg with libx264 found. Tried: ' + ', '.join(str(c) for c in cands))


ap = argparse.ArgumentParser()
ap.add_argument('--jobs', type=int, default=6)
ap.add_argument('--out', default='out/VIEWS')
A = ap.parse_args()

OBJS = ['ancient_lady', 'plane_waves', 'teapot_crack', 'hand_rorschach', 'skull_lava',
        'monster_lava_2', 'monster_rainbow', 'ancient_statue_clay', 'eagle_blackness',
        'mushroom_glow', 'alien_glow', 'ancient_lady_crack', 'spot_raurshaw']
NFR = {'ancient_lady': 135}

VIEWS = [('train',    'training view',            True),
         ('orbit360', '360° turntable',      False),
         ('orbit720', '720° turntable',      False),
         ('diagA',    'yaw 45° · elev +25°',   False),
         ('diagB',    'yaw 135° · elev −20°', False),
         ('diagC',    'yaw 225° · elev +30°',  False)]


def ready(o, v, n):
    for arm in ('r27', 'mcfm'):
        d = E / 'out' / f'view_{o}_{arm}_{v}' / 'frames'
        if not d.is_dir() or len(list(d.glob('*.png'))) < n:
            return False
    return True


def already(o, v, n):
    p = E / A.out / f'{o}_{v}_GT_frozen_r27_mcfm.mp4'
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        got = int(subprocess.run(
            [ffprobe_bin(), '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(p)],
            capture_output=True, text=True, check=True).stdout.strip())
        return got == n
    except Exception:
        return False


def build(t):
    o, v, label, sup, n = t
    cmd = [PY, str(E / 'build_view_panel.py'), '--obj', o, '--view', v,
           '--n', str(n), '--label', label, '--out', A.out]
    if sup:
        cmd.append('--supervised')
    r = subprocess.run(cmd, capture_output=True, text=True)
    tag = f'{o} {v}'
    if r.returncode != 0:
        return f'FAIL  {tag}: {(r.stdout + r.stderr).strip().splitlines()[-1][:120]}'
    return f'ok    {tag}  {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}'


def main():
    todo, skip_np, skip_done = [], 0, 0
    for o in OBJS:
        n = NFR.get(o, 150)
        for v, label, sup in VIEWS:
            if not ready(o, v, n):
                skip_np += 1
            elif already(o, v, n):
                skip_done += 1
            else:
                todo.append((o, v, label, sup, n))
    print(f'{len(todo)} to build, {skip_done} already done, {skip_np} not rendered yet', flush=True)
    with ThreadPoolExecutor(max_workers=A.jobs) as ex:
        for line in ex.map(build, todo):
            print(line, flush=True)
    built = len(list((E / A.out).glob('*.mp4'))) if (E / A.out).is_dir() else 0
    print(f'\n[DONE] {built} videos in {A.out}')


if __name__ == '__main__':
    main()
