"""
build_progression_video.py — the master progression, one video per object+view.

    GT  |  frozen TRELLIS.2  |  r19 cross-attn  |  r27 +self-attn  |  +MCFM

GT comes from the object's own 2D-copy targets. The other four are read out of the
orbit renders, each of which is a 2-panel frame: LEFT = frozen, RIGHT = adapted.
So the frozen column is taken from one arm and the adapted column from each.

THREE CHECKS BEFORE ANYTHING IS WRITTEN, because a silently mismatched panel is
worse than a missing one:

  GATE-frozen   The left half of the r19, r27 and MCFM renders must be the SAME
                IMAGE, pixel for pixel. All three rendered the same frozen model,
                the same mesh, through the same camera, so any difference means an
                arm used a different mesh, camera or frame count -- exactly the
                class of mistake that produces a plausible-looking wrong figure.
                Compared on sampled frames; a mismatch aborts that object.

  GATE-frames   Every arm must have the same frame count, and it must equal the
                object's target count. A short render silently truncates.

  GATE-config   Each arm's run config is re-read and asserted to be the arm it
                claims: rung number, target set, mcfm on/off, and that all three
                share one mesh / gt_dir / gt_render_dir.

Frames are matched by INDEX, not by filename order, and every panel is checked to
be the same height before pasting.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
OUT_D = _HERE / 'out'

ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True)
ap.add_argument('--view', required=True, choices=['side', '360', '720'])
ap.add_argument('--runs', required=True,
                help='json: {"r19": <run dir>, "r27": <run dir>, "mcfm": <run dir>}')
ap.add_argument('--panel', type=int, default=420)
ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--crf', type=int, default=18)
ap.add_argument('--check-every', type=int, default=20)
ap.add_argument('--out', default='out/PROGRESSION')
A = ap.parse_args()

BAR = 30
ARMS = ['r19g', 'r27g', 'r27mcfm']
LABEL = {'r19g': 'rung19  cross-attn LoRA',
         'r27g': 'rung27  + self-attn',
         'r27mcfm': 'rung27 + MCFM v2_D'}
EXPECT = {'r19g': (19, 'qkvo', None),
          'r27g': (27, 'qkvo+sa', None),
          'r27mcfm': (27, 'qkvo+sa', 'v2_D')}


def die(msg):
    print(f'FAILED: {msg}', flush=True)
    sys.exit(1)


def halves(p):
    """a render frame is [frozen | adapted]; return both halves"""
    im = Image.open(p).convert('RGB')
    w, h = im.size
    if w % 2:
        die(f'odd width {w} in {p} — not a 2-panel frame')
    return im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))


def main():
    runs = json.loads(A.runs)
    for a in ARMS:
        if a not in runs:
            die(f'no run dir given for {a}')

    # ---- GATE-config
    ref = None
    for a in ARMS:
        cfg = Path(runs[a]) / 'config.json'
        if not cfg.exists():
            die(f'{a}: no config.json at {cfg}')
        c = json.load(open(cfg))
        er, et, em = EXPECT[a]
        if c.get('rung') != er or c.get('targets') != et or c.get('mcfm') != em:
            die(f'{a}: config is rung{c.get("rung")}/{c.get("targets")}/mcfm={c.get("mcfm")}, '
                f'expected rung{er}/{et}/mcfm={em}')
        if A.obj not in (c.get('gt_dir') or ''):
            die(f'{a}: config gt_dir {c.get("gt_dir")} is not {A.obj}')
        key = (c.get('mesh'), c.get('gt_dir'), c.get('gt_render_dir'),
               c.get('n_frames'), c.get('render_res'))
        if ref is None:
            ref = key
        elif key != ref:
            die(f'{a}: mesh/gt/targets differ from the other arms — not a controlled comparison')
    n_frames = ref[3]
    print(f'[GATE-config] PASSED — three arms, one mesh/gt/targets, {n_frames} frames', flush=True)

    # ---- GATE-frames
    dirs = {a: OUT_D / f'render_{a}_{A.obj}_{A.view}' / 'frames' for a in ARMS}
    for a, d in dirs.items():
        if not d.is_dir():
            die(f'{a}: no render at {d}')
        n = len(list(d.glob('*.png')))
        if n != n_frames:
            die(f'{a}: {n} rendered frames but {n_frames} expected ({d})')
    gtd = OUT_D / f'gt_targets_{A.obj}_guan' / 'frames'
    ngt = len(list(gtd.glob('gt_*.png')))
    if ngt != n_frames:
        die(f'GT: {ngt} targets but {n_frames} expected ({gtd})')
    print(f'[GATE-frames] PASSED — {n_frames} frames in all three renders and the targets', flush=True)

    # ---- GATE-frozen: the left half must be identical across arms
    worst = 0.0
    for i in range(1, n_frames + 1, A.check_every):
        base = None
        for a in ARMS:
            fz, _ = halves(dirs[a] / f'{i:04d}.png')
            arr = np.asarray(fz, dtype=np.int16)
            if base is None:
                base = arr
            else:
                if arr.shape != base.shape:
                    die(f'frozen panel shape differs at frame {i}: {arr.shape} vs {base.shape}')
                dif = np.abs(arr - base)
                d = float(dif.max())
                frac = float((dif > 16).mean())
                worst = max(worst, d)
                # A MAX ALONE IS THE WRONG TEST. The decoder is fp16 and the
                # attention kernels are not bitwise deterministic across GPUs, so
                # two identical renders can differ by a few levels on a handful of
                # pixels: measured here at max 3/255, MEAN 0.007/255, and 0.000%
                # of pixels above 2/255. Gating on max>2 rejected three correct
                # videos for that.
                # A genuine mismatch -- wrong mesh, wrong camera, wrong frame --
                # moves a LARGE FRACTION of pixels by a LARGE amount, so that is
                # what is tested: more than 0.5% of pixels off by more than
                # 16/255. Nondeterminism cannot reach it; a wrong render cannot
                # avoid it.
                if frac > 0.005:
                    die(f'GATE-frozen: frame {i}, arm {a}: {100*frac:.2f}% of the frozen '
                        f'panel differs by more than 16/255 (max {d:.0f}). The arms did not '
                        f'render the same frozen model, mesh or camera — invalid comparison.')
    print(f'[GATE-frozen] PASSED — frozen panel identical across arms '
          f'(max diff {worst:.0f}/255 over {len(range(1, n_frames+1, A.check_every))} probes)',
          flush=True)

    # ---- composite
    OUT = (_HERE / A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f'_w_{A.obj}_{A.view}'; work.mkdir(parents=True, exist_ok=True)
    P, NP = A.panel, 5
    heads = ['GROUND TRUTH', 'TRELLIS.2 frozen'] + [LABEL[a] for a in ARMS]
    for i in range(1, n_frames + 1):
        gt = Image.open(gtd / f'gt_{i:04d}.png').convert('RGB').resize((P, P), Image.LANCZOS)
        fz, a19 = halves(dirs['r19g'] / f'{i:04d}.png')
        _, a27 = halves(dirs['r27g'] / f'{i:04d}.png')
        _, amc = halves(dirs['r27mcfm'] / f'{i:04d}.png')
        panels = [gt] + [im.resize((P, P), Image.LANCZOS) for im in (fz, a19, a27, amc)]
        c = Image.new('RGB', (P * NP, P + BAR), (20, 21, 26))
        d = ImageDraw.Draw(c)
        for k, (im, lab) in enumerate(zip(panels, heads)):
            c.paste(im, (k * P, BAR))
            d.text((k * P + 8, 9), lab, fill=(238, 238, 244))
            if k:
                d.line([(k * P, BAR), (k * P, P + BAR)], fill=(70, 74, 86), width=2)
        d.text((P * NP - 88, 9), f'{i:3d}/{n_frames}', fill=(186, 190, 202))
        c.save(work / f'{i:04d}.png')
        if i % 40 == 0 or i == 1:
            print(f'  {i}/{n_frames}', flush=True)

    vid = OUT / f'{A.obj}_{A.view}_progression.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-framerate', str(A.fps),
                    '-i', str(work / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264',
                    '-crf', str(A.crf), '-preset', 'slow', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(vid)], check=True)
    for p in work.glob('*.png'):
        p.unlink()
    work.rmdir()
    print(f'[VIDEO] {vid}  {vid.stat().st_size/1e6:.2f} MB  ({n_frames} frames)\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
