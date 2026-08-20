"""
build_fixview_panel.py — fixed-camera comparison, four azimuths, per object.

    GT | frozen TRELLIS.2 | rung27 | rung27 + MCFM

WHAT THE GT PANEL IS AT AN UNSEEN VIEW
  The Kling video is ONE camera, so no ground truth exists at 90/180/270. The GT
  column therefore shows the training-view video frame at every azimuth, and it
  is LABELLED as such: at yaw 0 it reads "GROUND TRUTH · training view", and at
  the other three "GT · training view (no GT at this angle)".

  It is kept because it is the texture reference a reader needs in shot -- what
  the lava is SUPPOSED to look like -- while the label prevents it being read as
  ground truth from that angle, which we do not have. The alternative, an empty
  column, tells the reader less and looks like an omission.

  The camera is PINNED for the whole sequence, so every change within a video is
  texture, never viewpoint. That is the entire point: in a turntable, viewpoint
  change swamps the temporal instability this is meant to expose.
"""
import argparse, json, subprocess, sys
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
ap.add_argument('--yaw', required=True, type=int)
ap.add_argument('--n', required=True, type=int)
ap.add_argument('--panel', type=int, default=400)
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=18)
ap.add_argument('--out', default='out/FIXVIEW')
A = ap.parse_args()
BAR = 32


def half(p, right):
    im = Image.open(p).convert('RGB'); w, h = im.size
    return im.crop((w // 2, BAR - 4, w, h)) if right else im.crop((0, BAR - 4, w // 2, h))


def main():
    d27 = _HERE / 'out' / f'fixview_{A.obj}_r27_yaw{A.yaw}' / 'frames'
    dmc = _HERE / 'out' / f'fixview_{A.obj}_mcfm_yaw{A.yaw}' / 'frames'
    gtd = T2 / 'data' / A.obj / 'frames_from_video'
    for p in (d27, dmc, gtd):
        if not p.is_dir():
            sys.exit(f'FAILED: missing {p}')
    for p in (d27, dmc):
        got = len(list(p.glob('*.png')))
        if got != A.n:
            sys.exit(f'FAILED: {p} has {got} frames, expected {A.n}')

    sup = (A.yaw == 0)
    gtlab = 'GROUND TRUTH  ·  training view' if sup else 'GT  ·  training view (no GT at this angle)'
    heads = [gtlab, f'frozen TRELLIS.2  ·  {A.yaw}°', f'rung27  ·  {A.yaw}°', f'rung27 + MCFM  ·  {A.yaw}°']
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_w_{A.obj}_{A.yaw}'; work.mkdir(parents=True, exist_ok=True)
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
            dr.text((k * P + 9, 10), lab, fill=(150, 156, 168) if (k == 0 and not sup) else (238, 238, 244))
            if k:
                dr.line([(k * P, BAR), (k * P, P + BAR)], fill=(70, 74, 86), width=2)
        dr.text((P * 4 - 86, 10), f'{i:3d}/{A.n}', fill=(186, 190, 202))
        c.save(work / f'{i:04d}.png')

    vid = OUT / f'{A.obj}_yaw{A.yaw}_GT_frozen_r27_mcfm.mp4'
    subprocess.run([ffmpeg_bin(), '-nostdin', '-v', 'error', '-y', '-framerate', str(A.fps),
                    '-i', str(work / '%04d.png'), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', str(A.crf), '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(vid)], check=True)
    for p in work.glob('*.png'):
        p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid.name}  {vid.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
