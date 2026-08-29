"""video_metrics.py -- flicker / acceleration / drift on RENDERED FRAMES.

WHY THIS EXISTS
  The texel metrics in render_rung27_orbit.py are defined on the PBR voxel field,
  which is the right measurement for us and impossible for a baseline that never
  produces a 3D field. For the comparison table every method must be scored the same
  way, so the same three quantities are computed here on the rendered video instead.

SAME ESTIMATOR AS THE TEXEL VERSION, DIFFERENT SUPPORT
  F = mean_t <|I_t - I_{t-1}|>      A = mean_t <|I_t - 2I_{t-1} + I_{t-2}|>
  D = <|I_T - I_1|>                 <.> = mean over masked pixels AND channels
  Identical to eqs (1)-(3) of the metrics section with C_t -> I_t. Values are NOT
  numerically comparable to the texel ones -- different support, different units --
  so a video row and a texel row must never share a column.

THE CAMERA MUST BE FIXED
  A moving camera makes every frame differ from the last whether or not the texture
  changed, which is the confound texel space was chosen to avoid. Frames must come
  from a fixed viewpoint. This is asserted only by convention -- nothing in a folder
  of PNGs records the camera -- so the caller is responsible, and --expect-frames
  exists to catch the common case of pointing at an orbit render by mistake.

THE MASK MUST BE SHARED ACROSS METHODS
  Background is ~89% of the pixels and near-static for everyone, so an unmasked score
  is dominated by pixels no method is being judged on and every method flattens toward
  zero. Worse, a per-method threshold gives each method a different denominator. Pass
  ONE --mask for all methods on an object. --bg-thresh is a fallback for a
  single-method sanity check, never for a table.

MEMORY
  Frames are streamed through a three-frame rolling window rather than stacked: 150
  frames at 512x512x3 float32 is 472 MB stacked, and at 960x960 it is 1.7 GB.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True, help='directory of rendered frames')
ap.add_argument('--glob', default='frame_*.png',
                help="filename pattern; run_baseline.py writes frame_####.png, "
                     "render_rung27_orbit.py writes ####.png")
ap.add_argument('--panel', type=int, default=None,
                help='crop panel J out of a side-by-side composite (0-based). '
                     'render_rung27_orbit.py writes [frozen | ours] composites.')
ap.add_argument('--n-panels', type=int, default=1)
ap.add_argument('--label-h', type=int, default=28,
                help='height of the label bar composites carry on top '
                     '(render_rung27_orbit.py: LAB = 28)')
ap.add_argument('--mask', default=None, help='.npy boolean HxW, shared by all methods')
ap.add_argument('--bg-thresh', type=int, default=None,
                help='fallback mask: foreground = any channel < thresh. '
                     'Single-method sanity checks only -- never for a table.')
ap.add_argument('--expect-frames', type=int, default=None,
                help='fail if the directory does not hold exactly this many frames')
ap.add_argument('--tag', default=None)
ap.add_argument('--out', required=True)
A = ap.parse_args()

files = sorted(Path(A.frames).glob(A.glob))
if not files:
    raise SystemExit(f'no frames matching {A.glob!r} under {A.frames}')
if A.expect_frames is not None and len(files) != A.expect_frames:
    raise SystemExit(f'expected {A.expect_frames} frames, found {len(files)} '
                     f'-- refusing to measure a truncated sequence')


def load(p):
    """-> float32 HxWx3 in [0,1], panel-cropped if asked."""
    a = np.asarray(Image.open(p).convert('RGB'))
    if A.panel is not None:
        H, W = a.shape[:2]
        side = H - A.label_h
        # GATE-layout. render_rung27_orbit.py builds (res+LAB, res*n_panels) and
        # pastes panel j at (j*res, LAB). If that ever changes, the crop would
        # silently measure the wrong pixels, so prove the arithmetic instead.
        if side * A.n_panels != W:
            raise SystemExit(
                f'GATE-layout FAILED on {p.name}: (H-label_h)*n_panels = '
                f'({H}-{A.label_h})*{A.n_panels} = {side*A.n_panels} != W = {W}. '
                f'--n-panels/--label-h do not describe this image.')
        if not 0 <= A.panel < A.n_panels:
            raise SystemExit(f'--panel {A.panel} outside 0..{A.n_panels-1}')
        a = a[A.label_h:A.label_h + side, A.panel * side:(A.panel + 1) * side]
    return a.astype(np.float32) / 255.0


first = load(files[0])
H, W = first.shape[:2]

if A.mask:
    m = np.load(A.mask)
    if m.dtype != bool:
        m = m.astype(bool)
    if m.shape != (H, W):
        raise SystemExit(f'mask is {m.shape}, frames are {(H, W)} -- a mask from a '
                         f'different resolution would score the wrong pixels')
    mask_src = A.mask
elif A.bg_thresh is not None:
    # union over the first and last frame so a texture that darkens or brightens
    # mid-sequence cannot shrink the support underneath the metric.
    last = load(files[-1])
    m = ((first * 255 < A.bg_thresh).any(-1) | (last * 255 < A.bg_thresh).any(-1))
    mask_src = f'bg-thresh {A.bg_thresh} (fallback, not table-grade)'
else:
    m = np.ones((H, W), bool)
    mask_src = 'none (whole frame -- background will dominate)'

npx = int(m.sum())
if npx == 0:
    raise SystemExit('mask selects zero pixels')

flick, jerk = [], []
prev = prev2 = None
for p in files:
    cur = load(p)
    if cur.shape[:2] != (H, W):
        raise SystemExit(f'{p.name} is {cur.shape[:2]}, expected {(H, W)} -- frames '
                         f'of mixed size cannot be differenced')
    if prev is not None:
        flick.append(float(np.abs(cur[m] - prev[m]).mean()))
    if prev2 is not None:
        jerk.append(float(np.abs(cur[m] - 2 * prev[m] + prev2[m]).mean()))
    prev2, prev = prev, cur

if len(flick) < 2:
    raise SystemExit('need at least 3 frames')

rec = {
    'tag': A.tag or Path(A.frames).name,
    'frames_dir': str(Path(A.frames).resolve()),
    'space': 'video',                       # never mix with a texel row
    'n_frames': len(files),
    'resolution': [H, W],
    'mask_pixels': npx,
    'mask_frac': npx / (H * W),
    'mask_source': mask_src,
    'panel': A.panel,
    'video_flicker': float(np.mean(flick)),
    'video_accel': float(np.mean(jerk)),
    'video_drift': float(np.abs(prev[m] - first[m]).mean()),
    'flicker_per_frame': flick,
    'accel_per_frame': jerk,
}
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
Path(A.out).write_text(json.dumps(rec, indent=1))
print(f"[VIDEO] {rec['tag']}  frames {rec['n_frames']}  {H}x{W}  "
      f"mask {npx:,} px ({100*rec['mask_frac']:.1f}%)  src={mask_src}")
print(f"[VIDEO] flicker {rec['video_flicker']:.6f}   accel {rec['video_accel']:.6f}   "
      f"drift {rec['video_drift']:.6f}")
print(f"[VIDEO] -> {A.out}")
