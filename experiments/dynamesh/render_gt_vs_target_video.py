"""
render_gt_vs_target_video.py — GT video frame vs the 2D copy, every frame.

Two panels, side by side, for the whole sequence:

    1  GROUND TRUTH     the raw video frame the generative model produced
    2  2D COPY          Option B — that frame's colour resampled onto OUR mesh's
                        silhouette in screen space. What the loss actually uses.
    3  BACKPROJECTION   Option C — the same colour pushed out to the mesh's
                        VERTICES and re-rendered. Goes through the 3D surface
                        rather than staying in screen space.

Watching this as a video is the only way to see the thing that matters — whether
the copy tracks the video's texture faithfully across the sequence, and where the
two silhouettes drift apart as the video model reshapes the object over time. A
single still cannot show either.

CPU only. No model, no GPU.
"""
import argparse, json, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')

# object -> (video frames dir, gt_targets tag, n_frames)
OBJ = {
 'spot_lava':          ('data/spot_lava/frames_from_video',        'gt_targets_spot_lava',        150),
 'spot_star':          ('data/spot_star/frames_from_video',        'gt_targets_spot_star',        150),
 'teapot_lava2':       ('data/teapot_lava2/frames_from_video',     'gt_targets_teapot_lava2',     150),
 'teapot_960':         ('data/teapot/frames_from_video',           'gt_targets_teapot_960',       150),
 'teapot_porcelain':   ('data/teapot_porcelain/frames_from_video', 'gt_targets_teapot_porcelain', 121),
 'horse_metal':        ('data/horse_metal/frames_from_video',      'gt_targets_horse_metal',      121),
 'penguin_circuits':   ('data/penguin_circuits/frames_from_video', 'gt_targets_penguin_circuits', 121),
 'whale_spots':        ('data/whale_spots/frames_from_video',      'gt_targets_whale_spots',      121),
 'pumpkin_rot':        ('data/pumpkin_rot/frames_from_video',      'gt_targets_pumpkin_rot',      121),
 'teapot_ceramic_crack': ('data/teapot_ceramic_crack/frames_from_video',
                                                                  'gt_targets_teapot_ceramic_crack', 121),
}

ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True, choices=sorted(OBJ))
ap.add_argument('--panel', type=int, default=440, help='pixels per panel')
ap.add_argument('--no-backproj', action='store_true',
                help='2 panels only, if the backprojection stage has not run')
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=17)
ap.add_argument('--suffix', default='', help='target/backproj tag suffix, e.g. _guan')
ap.add_argument('--out', default='out/GT_VS_TARGET')
A = ap.parse_args()

BAR = 30


def main():
    gd, tag, n = OBJ[A.obj]
    frames = T2 / gd
    tdir = _HERE / 'out' / f'{tag}{A.suffix}' / 'frames'
    bdir = _HERE / 'out' / f'gt_rendered_{A.obj}{A.suffix}' / 'frames'
    use_bp = (not A.no_backproj) and bdir.exists()
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_f_{A.obj}'; work.mkdir(parents=True, exist_ok=True)

    assert tdir.exists(), f'no targets: {tdir}'
    P = A.panel
    NP = 3 if use_bp else 2
    print(f'{A.obj}: {n} frames, {NP} x {P}px panels', flush=True)
    print(f'  1 GT          {frames}')
    print(f'  2 2D copy     {tdir}')
    print(f'  3 backproj    {bdir if use_bp else "(missing — 2 panels only)"}', flush=True)

    for i in range(1, n + 1):
        v = Image.open(frames / f'frame_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)
        t = Image.open(tdir / f'gt_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)
        panels = [(v, 'GROUND TRUTH  -  video frame'),
                  (t, '2D COPY  -  option B, screen space')]
        if use_bp:
            b = Image.open(bdir / f'gt_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)
            panels.append((b, 'BACKPROJECTION  -  option C, via the surface'))
        c = Image.new('RGB', (P * NP, P + BAR), (22, 23, 28))
        d = ImageDraw.Draw(c)
        for k, (im, lab) in enumerate(panels):
            c.paste(im, (k * P, BAR))
            d.text((k * P + 10, 9), lab, fill=(238, 238, 244))
            if k:
                d.line([(k * P, BAR), (k * P, P + BAR)], fill=(70, 74, 86), width=2)
        d.text((P * NP - 92, 9), f'{i:3d} / {n}', fill=(190, 194, 206))
        c.save(work / f'{i:04d}.png')
        if i % 40 == 0 or i == 1:
            print(f'  {i}/{n}', flush=True)

    vid = OUT / (f'{A.obj}{A.suffix}_GT_2Dcopy_backproj.mp4' if use_bp
             else f'{A.obj}{A.suffix}_GT_vs_2Dcopy.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-framerate', str(A.fps),
                    '-i', str(work / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', str(A.crf), '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(vid)], check=True)
    for p in work.glob('*.png'):
        p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid}  {vid.stat().st_size/1e6:.2f} MB  ({n} frames)\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
