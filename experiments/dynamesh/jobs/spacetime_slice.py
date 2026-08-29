"""spacetime_slice.py — make temporal coherence visible.

THE PROBLEM. Flicker and acceleration are numbers about what happens BETWEEN frames,
and no single frame can show them. Two sequences with identical per-frame appearance,
one smooth and one jittering, look the same in any still figure.

THE FIGURE. Take one fixed scanline through the object and stack it over all T
frames: x is position along the surface, y is time. A texture evolving smoothly draws
continuous streaks; one that jitters draws vertical noise. This is the standard
space-time slice used in video super-resolution, and it makes the metric legible.

The scanline is chosen ONCE, from the object's silhouette in the middle frame, and
reused for every arm -- otherwise the arms would be sliced at different places and the
comparison would be of scanlines, not of methods.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True)
ap.add_argument('--arms', nargs='+', required=True, help='panel dir suffixes, e.g. rung27 mcfm')
ap.add_argument('--yaw', default='y0')
# view_<obj>_<arm>_u0 covers all 14 arms for every object; panel_<obj>_<arm>_y0 only
# exists for the four figure objects. Same 1036x546 layout either way.
ap.add_argument('--pattern', default='out/panel_{obj}_{arm}_{yaw}/frames/*.png')
ap.add_argument('--out', required=True)
ap.add_argument('--stretch', type=int, default=3, help='vertical px per frame')
ap.add_argument('--label', action='append', default=[],
                help='NAME=TEXT, e.g. rung27="rung27  0.0061 / 0.0087"')
A = ap.parse_args()
A.labels = dict(l.split('=', 1) for l in A.label) if A.label else {}
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
S, Y0 = 518, 28                      # panel geometry: [frozen|ours] under a 28px label

def frames(arm):
    return sorted(glob.glob(str(E / A.pattern.format(obj=A.obj, arm=arm, yaw=A.yaw))))

ref = frames(A.arms[0])
if not ref:
    sys.exit(f'no frames for {A.arms[0]}')
mid = np.asarray(Image.open(ref[len(ref) // 2]).convert('RGB'))[Y0:Y0 + S, S:2 * S]
# widest row of the silhouette: most surface, so the most texture to judge
fg = (mid.min(axis=2) < 240)
row = int(np.argmax(fg.sum(axis=1)))
x0, x1 = int(np.argmax(fg[row])), int(S - np.argmax(fg[row][::-1]))
print(f'scanline row={row} cols={x0}..{x1} ({x1-x0}px of surface)', flush=True)

def slice_for(arm, half):
    fs = frames(arm)
    if not fs: return None
    out = np.zeros((len(fs), x1 - x0, 3), np.uint8)
    for i, f in enumerate(fs):
        im = np.asarray(Image.open(f).convert('RGB'))
        xo = 0 if half == 'L' else S
        out[i] = im[Y0 + row, xo + x0: xo + x1]
    return np.repeat(out, A.stretch, axis=0)

cols = [('frozen TRELLIS.2', slice_for(A.arms[0], 'L'))]
for a in A.arms:
    cols.append((a, slice_for(a, 'R')))
cols = [(n, c) for n, c in cols if c is not None]
h = min(c.shape[0] for _, c in cols)
pad, lab = 14, 30
W = sum(c.shape[1] for _, c in cols) + pad * (len(cols) - 1)
canvas = np.full((h + lab, W, 3), 255, np.uint8)
x = 0
spans = []
for n, c in cols:
    canvas[lab:lab + h, x:x + c.shape[1]] = c[:h]
    spans.append((n, x, c.shape[1]))
    x += c.shape[1] + pad
img = Image.fromarray(canvas)
# Labels carry the measured numbers, so the figure and the table cannot drift apart:
# a reader can check the claim the picture is making against the row it came from.
from PIL import ImageDraw, ImageFont
dr = ImageDraw.Draw(img)
try:
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
except Exception:
    fnt = ImageFont.load_default()
for n, x0_, w_ in spans:
    t = A.labels.get(n, n) if A.labels else n
    tw = dr.textlength(t, font=fnt)
    dr.text((x0_ + max(0, (w_ - tw) / 2), 7), t, fill=(20, 23, 28), font=fnt)
img.save(A.out)
print(f'wrote {A.out}  {canvas.shape[1]}x{canvas.shape[0]}  ({len(cols)} columns)')
print('columns:', ', '.join(n for n, _ in cols))
