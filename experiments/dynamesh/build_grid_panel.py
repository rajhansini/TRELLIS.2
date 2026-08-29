"""
build_grid_panel.py — GT + frozen + every rung arm, tiled over MULTIPLE ROWS.

build_view_panel.py is hardcoded to four columns (GT | frozen | rung27 | rung27+MCFM).
The full ladder is 14 arms, which is 16 tiles with GT and frozen — far too wide for
one strip, so this lays them out on a grid and captions each tile.

WHERE THE TILES COME FROM
  render_arm.py writes one image per frame that is already `frozen | adapted` side by
  side, exactly like render_rung27_orbit.py. So:
      frozen  = LEFT half of ANY arm's render
      arm     = RIGHT half of that arm's render
  Taking frozen from an arm we already rendered rather than rendering it a 17th time
  guarantees it is the same frozen model, same camera, same frame indexing. It cannot
  drift out of step with the arms it is being compared against.

WHAT THE GT TILE IS AWAY FROM THE TRAINING VIEW
  The Kling video is ONE camera. There is no ground truth at any other view, so at an
  unseen view the GT tile shows the training-view frame and SAYS SO in its caption,
  dimmed. It is kept because a reader needs the texture reference in shot — what the
  effect is supposed to be doing at this instant — while the caption stops it being
  read as ground truth from an angle we do not have.

MISSING ARMS ARE DRAWN, NOT SKIPPED
  An object part-way through the ladder would otherwise silently produce a smaller
  grid, and a 12-tile panel looks just as finished as a 16-tile one. Absent arms get
  a visible "not trained" placeholder so the gap is legible.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
BAR = 30

# ---------------------------------------------------------------- ffmpeg lookup
# THREE ffmpegs on this cluster and only two can do the job:
#   /usr/bin/ffmpeg                  libx264 YES -- but LOGIN NODE ONLY
#   conda_envs/trellis2/bin/ffmpeg   libx264 NO  -- openh264 only, exits 8
#   conda_envs/richards_distillation libx264 YES -- shared storage, all nodes
# Probe for the ENCODER, not the binary, and cache the answer.
_FFMPEG = None


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
    raise SystemExit('FAILED: no ffmpeg with libx264. Tried: ' + ', '.join(str(c) for c in cands))


def ffprobe_bin():
    return str(Path(ffmpeg_bin()).with_name('ffprobe'))


# The ladder, in the order a reader should scan it: baseline first, then the KL
# arms, then the temporal ones. 32/34 take no MCFM -- --context-window and --mcfm
# are mutually exclusive, so those cells do not exist and are not drawn as gaps.
ARMS = [
    ('27', 'run27'), ('27m', 'run27 + MCFM'),
    ('28', 'run28  KL self'), ('28m', 'run28 + MCFM'),
    ('29', 'run29  KL cross'), ('29m', 'run29 + MCFM'),
    ('30', 'run30  KL both'), ('30m', 'run30 + MCFM'),
    ('31', 'run31  dual branch'), ('31m', 'run31 + MCFM'),
    ('32', 'run32  wide context'),
    ('33', 'run33  dual + KL'), ('33m', 'run33 + MCFM'),
    ('34', 'run34  wide + KL'),
]
# Keys match render_arm.sbatch's TAG convention exactly: view_<obj>_<arm>_<view>,
# arm in {27,27m,28,28m,...,34} with NO 'r' prefix. A stray leading 'r' here once
# silently matched only the unrelated pre-existing view_<obj>_r27_<view> /
# view_<obj>_mcfm_<view> dirs from the OLD build_view_panel.py pipeline and drew
# every other arm as "not trained" despite 14/14 being complete on disk.

ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True)
ap.add_argument('--view', required=True, help='view key, e.g. train / diagA / diagB / diagC')
ap.add_argument('--n', type=int, required=True)
ap.add_argument('--label', required=True, help='human camera description')
ap.add_argument('--supervised', action='store_true',
                help='set ONLY for the training view, where the GT tile is real')
ap.add_argument('--tile', type=int, default=260)
ap.add_argument('--cols', type=int, default=4)
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=18)
ap.add_argument('--out', default='out/GRID')
A = ap.parse_args()


def arm_dir(arm):
    return _HERE / 'out' / f'view_{A.obj}_{arm}_{A.view}' / 'frames'


def half(p, right):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    return im.crop((w // 2, BAR - 4, w, h)) if right else im.crop((0, BAR - 4, w // 2, h))


def main():
    gtd = T2 / 'data' / A.obj / 'frames_from_video'
    if not gtd.is_dir():
        sys.exit(f'FAILED: missing {gtd}')

    present = [(k, lab) for k, lab in ARMS
               if arm_dir(k).is_dir() and len(list(arm_dir(k).glob('*.png'))) >= A.n]
    if not present:
        sys.exit(f'FAILED: no arm of {A.obj} has {A.n} rendered frames at view {A.view}')

    # frozen comes from whichever arm rendered first -- same frozen model either way
    frozen_src = arm_dir(present[0][0])
    missing = [k for k, _ in ARMS if k not in {p[0] for p in present}]
    if missing:
        print(f'[GRID] {A.obj} {A.view}: {len(present)}/{len(ARMS)} arms rendered, '
              f'missing {missing} — drawn as placeholders', flush=True)

    tiles = [('__gt__', 'GROUND TRUTH' if A.supervised else 'GT · training view'),
             ('__fz__', 'frozen TRELLIS.2')] + ARMS
    ncol = A.cols
    nrow = (len(tiles) + ncol - 1) // ncol
    P = A.tile
    W, H = ncol * P, nrow * (P + BAR)

    OUT = (_HERE / A.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_w_{A.obj}_{A.view}'
    work.mkdir(parents=True, exist_ok=True)
    have = {k for k, _ in present}

    for i in range(1, A.n + 1):
        canvas = Image.new('RGB', (W, H + BAR), (20, 21, 26))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 9), f'{A.obj}   ·   {A.label}   ·   frame {i:3d}/{A.n}',
                fill=(238, 238, 244))
        for idx, (key, lab) in enumerate(tiles):
            r, c = divmod(idx, ncol)
            x, y = c * P, BAR + r * (P + BAR)
            if key == '__gt__':
                im = Image.open(gtd / f'frame_{i:04d}.png').convert('RGB')
                im = im.resize((P, P), Image.LANCZOS)
                fill = (238, 238, 244) if A.supervised else (150, 156, 168)
            elif key == '__fz__':
                im = half(frozen_src / f'{i:04d}.png', False).resize((P, P), Image.LANCZOS)
                fill = (238, 238, 244)
            elif key in have:
                im = half(arm_dir(key) / f'{i:04d}.png', True).resize((P, P), Image.LANCZOS)
                fill = (238, 238, 244)
            else:
                im = Image.new('RGB', (P, P), (34, 35, 42))
                d2 = ImageDraw.Draw(im)
                d2.text((P // 2 - 34, P // 2 - 6), 'not trained', fill=(110, 114, 126))
                fill = (110, 114, 126)
            canvas.paste(im, (x, y))
            dr.text((x + 7, y - 19), lab, fill=fill)
            if c:
                dr.line([(x, y), (x, y + P)], fill=(70, 74, 86), width=2)
        canvas.save(work / f'{i:04d}.png')

    vid = OUT / f'{A.obj}_{A.view}_grid.mp4'
    subprocess.run([ffmpeg_bin(), '-nostdin', '-v', 'error', '-y',
                    '-framerate', str(A.fps), '-start_number', '1',
                    '-i', str(work / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', str(A.crf), '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(vid)], check=True)

    # GATE-encode: the encode must carry every frame the panels were built from.
    # A short render silently yields a short video that still looks fine.
    n = int(subprocess.run([ffprobe_bin(), '-v', 'error', '-select_streams', 'v:0',
                            '-count_frames', '-show_entries', 'stream=nb_read_frames',
                            '-of', 'csv=p=0', str(vid)],
                           capture_output=True, text=True, check=True).stdout.strip())
    if n != A.n:
        sys.exit(f'FAILED: encoded {n} frames, expected {A.n} — not deleting {work}')

    for p in work.glob('*.png'):
        p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid.name}  {n} frames  {len(present)}/{len(ARMS)} arms  '
          f'{vid.stat().st_size / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
