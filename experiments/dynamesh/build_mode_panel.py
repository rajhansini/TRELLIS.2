"""
build_mode_panel.py — four-panel comparison of the three MCFM blend modes.

    GROUND TRUTH | temporal_only | spatial_then_temporal | joint

All three arms are rung27 trained identically; the ONLY difference is which blend
produced the conditioning. So any difference visible here is attributable to the
blend and nothing else.

The frozen column is deliberately absent: it is the same frozen model in all three
arms and would waste a quarter of the width on a constant. The comparison that
matters is mode against mode.
"""
import argparse, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
_FFMPEG = None


def ffmpeg_bin():
    global _FFMPEG
    if _FFMPEG: return _FFMPEG
    import shutil as _sh
    for c in ['/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin/ffmpeg',
              '/usr/bin/ffmpeg', _sh.which('ffmpeg')]:
        if not c or not Path(c).exists(): continue
        try:
            if 'libx264' in subprocess.run([c,'-hide_banner','-encoders'],
                    capture_output=True,text=True,timeout=30).stdout:
                _FFMPEG = c; return c
        except Exception: pass
    raise SystemExit('FAILED: no ffmpeg with libx264')


def ffprobe_bin():
    return str(Path(ffmpeg_bin()).with_name('ffprobe'))


ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True)
ap.add_argument('--view', required=True)
ap.add_argument('--n', required=True, type=int)
ap.add_argument('--label', required=True)
ap.add_argument('--supervised', action='store_true')
ap.add_argument('--panel', type=int, default=400)
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=18)
ap.add_argument('--out', default='out/MODES')
A = ap.parse_args()
BAR = 32
MODES = [('v2_D', 'temporal only'), ('st_D', 'spatial → temporal'), ('v3_D', 'joint')]


def right_half(p):
    """render_rung27_orbit writes frozen|adapted side by side; the adapted arm is
    the right half. Cropping rather than re-rendering keeps every panel on the
    same frame indexing."""
    im = Image.open(p).convert('RGB'); w, h = im.size
    return im.crop((w // 2, BAR - 4, w, h))


def main():
    dirs = {c: _HERE / 'out' / f'm3_{A.obj}_{c}_{A.view}' / 'frames' for c, _ in MODES}
    gtd = T2 / 'data' / A.obj / 'frames_from_video'
    for c, d in dirs.items():
        if not d.is_dir(): sys.exit(f'FAILED: missing {d}')
        got = len(list(d.glob('*.png')))
        if got != A.n: sys.exit(f'FAILED: {d} has {got}/{A.n} frames')
    if not gtd.is_dir(): sys.exit(f'FAILED: missing {gtd}')

    gtlab = ('GROUND TRUTH  ·  training view' if A.supervised
             else f'GT  ·  training view (no GT at {A.label})')
    heads = [gtlab] + [f'{lab}  ·  {A.label}' for _, lab in MODES]

    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_w_{A.obj}_{A.view}'; work.mkdir(parents=True, exist_ok=True)
    P = A.panel

    for i in range(1, A.n + 1):
        ims = [Image.open(gtd / f'frame_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)]
        ims += [right_half(dirs[c] / f'{i:04d}.png').resize((P, P), Image.LANCZOS) for c, _ in MODES]
        c_im = Image.new('RGB', (P * 4, P + BAR), (20, 21, 26))
        dr = ImageDraw.Draw(c_im)
        for k, (im, lab) in enumerate(zip(ims, heads)):
            c_im.paste(im, (k * P, BAR))
            dr.text((k * P + 9, 10), lab,
                    fill=(150, 156, 168) if (k == 0 and not A.supervised) else (238, 238, 244))
            if k: dr.line([(k * P, BAR), (k * P, P + BAR)], fill=(70, 74, 86), width=2)
        dr.text((P * 4 - 86, 10), f'{i:3d}/{A.n}', fill=(186, 190, 202))
        c_im.save(work / f'{i:04d}.png')

    vid = OUT / f'{A.obj}_{A.view}_GT_3modes.mp4'
    subprocess.run([ffmpeg_bin(), '-nostdin', '-v', 'error', '-y', '-framerate', str(A.fps),
                    '-start_number', '1', '-i', str(work / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', str(A.crf), '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(vid)], check=True)
    n = int(subprocess.run([ffprobe_bin(), '-v', 'error', '-select_streams', 'v:0',
                            '-count_frames', '-show_entries', 'stream=nb_read_frames',
                            '-of', 'csv=p=0', str(vid)],
                           capture_output=True, text=True, check=True).stdout.strip())
    if n != A.n: sys.exit(f'FAILED: encoded {n}/{A.n} frames — keeping {work}')
    for p in work.glob('*.png'): p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid.name}  {n} frames  {vid.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
