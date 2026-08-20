"""
build_view_panel.py — four-panel comparison at an arbitrary camera.

    GROUND TRUTH | frozen TRELLIS.2 | rung27 | rung27 + MCFM

Generalises build_fixview_panel.py, which only knew about the four canonical
azimuths at elevation 0. Here a view is (yaw, elev, turns), so the same builder
covers the training view, the 360 and 720 turntables, and the diagonal views that
sit off both the equator and the canonical azimuths.

WHERE THE FOUR PANELS COME FROM
  render_rung27_orbit.py writes one image per frame that is already
  frozen | adapted side by side. So the frozen column is the LEFT half of the
  rung27 render, and the two adapted columns are the RIGHT halves of the rung27
  and the rung27+MCFM renders. Taking frozen from the r27 render rather than
  rendering it a third time guarantees it is the same frozen model, same camera,
  same frame indexing — it cannot drift out of step with the arm it is compared to.

WHAT THE GT PANEL IS AWAY FROM THE TRAINING VIEW
  The Kling video is ONE camera. There is no ground truth at any other view, so
  the GT column shows the training-view video frame everywhere, and says so in its
  label — full brightness only at the training view, dimmed elsewhere. It is kept
  because it is the texture reference a reader needs in shot (what the lava is
  SUPPOSED to be doing at this instant) while the label stops it being read as
  ground truth from an angle we do not have. An empty column would tell the reader
  less and look like an omission.

  On the turntables the GT column additionally cannot follow the camera, which is
  the honest reason those videos are for judging coverage and seams rather than
  fidelity — the fidelity claim lives at the training view and in the PSNR table.
"""
import argparse, subprocess, sys
from pathlib import Path

from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

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
ap.add_argument('--obj', required=True)
ap.add_argument('--view', required=True, help='view key, e.g. train / orbit360 / diagA')
ap.add_argument('--n', required=True, type=int)
ap.add_argument('--label', required=True, help='human camera description for the headers')
ap.add_argument('--supervised', action='store_true',
                help='set ONLY for the training view, where the GT column is real')
ap.add_argument('--panel', type=int, default=400)
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=18)
ap.add_argument('--out', default='out/VIEWS')
A = ap.parse_args()
BAR = 32


def half(p, right):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    return im.crop((w // 2, BAR - 4, w, h)) if right else im.crop((0, BAR - 4, w // 2, h))


def main():
    d27 = _HERE / 'out' / f'view_{A.obj}_r27_{A.view}' / 'frames'
    dmc = _HERE / 'out' / f'view_{A.obj}_mcfm_{A.view}' / 'frames'
    gtd = T2 / 'data' / A.obj / 'frames_from_video'
    for p in (d27, dmc, gtd):
        if not p.is_dir():
            sys.exit(f'FAILED: missing {p}')

    # GATE-frames: a short render silently yields a short video that still looks
    # fine. spot_lava shipped at 27 of 150 frames once before this check existed.
    for p in (d27, dmc):
        got = len(list(p.glob('*.png')))
        if got != A.n:
            sys.exit(f'FAILED: {p} has {got} frames, expected {A.n}')

    gtlab = ('GROUND TRUTH  ·  training view' if A.supervised
             else f'GT  ·  training view (no GT at {A.label})')
    heads = [gtlab, f'frozen TRELLIS.2  ·  {A.label}',
             f'rung27  ·  {A.label}', f'rung27 + MCFM  ·  {A.label}']

    OUT = (_HERE / A.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_w_{A.obj}_{A.view}'
    work.mkdir(parents=True, exist_ok=True)
    P = A.panel

    for i in range(1, A.n + 1):
        gt = Image.open(gtd / f'frame_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)
        fz = half(d27 / f'{i:04d}.png', False).resize((P, P), Image.LANCZOS)
        a27 = half(d27 / f'{i:04d}.png', True).resize((P, P), Image.LANCZOS)
        amc = half(dmc / f'{i:04d}.png', True).resize((P, P), Image.LANCZOS)
        c = Image.new('RGB', (P * 4, P + BAR), (20, 21, 26))
        dr = ImageDraw.Draw(c)
        for k, (im, lab) in enumerate(zip([gt, fz, a27, amc], heads)):
            c.paste(im, (k * P, BAR))
            dr.text((k * P + 9, 10), lab,
                    fill=(150, 156, 168) if (k == 0 and not A.supervised) else (238, 238, 244))
            if k:
                dr.line([(k * P, BAR), (k * P, P + BAR)], fill=(70, 74, 86), width=2)
        dr.text((P * 4 - 86, 10), f'{i:3d}/{A.n}', fill=(186, 190, 202))
        c.save(work / f'{i:04d}.png')

    vid = OUT / f'{A.obj}_{A.view}_GT_frozen_r27_mcfm.mp4'
    subprocess.run([ffmpeg_bin(), '-nostdin', '-v', 'error', '-y', '-framerate', str(A.fps),
                    '-start_number', '1', '-i', str(work / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', str(A.crf), '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(vid)], check=True)

    # GATE-encode: the encode must carry every frame the panels were built from
    n = int(subprocess.run([ffprobe_bin(), '-v', 'error', '-select_streams', 'v:0',
                            '-count_frames', '-show_entries', 'stream=nb_read_frames',
                            '-of', 'csv=p=0', str(vid)],
                           capture_output=True, text=True, check=True).stdout.strip())
    if n != A.n:
        sys.exit(f'FAILED: encoded {n} frames, expected {A.n} — not deleting {work}')

    for p in work.glob('*.png'):
        p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid.name}  {n} frames  {vid.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
