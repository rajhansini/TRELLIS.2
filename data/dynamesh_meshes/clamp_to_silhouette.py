"""
clamp_to_silhouette.py — force a Kling clip back inside the mesh outline.

WHY THIS EXISTS AND NOT A BETTER PROMPT
  Kling puts glow, bolts and flames OUTSIDE the object no matter how the prompt is
  worded, because naming an emissive phenomenon is what makes it emit. That spill is
  fatal downstream: the fit optimises a texture on a FIXED mesh, so any pixel outside
  the silhouette has no surface to land on and turns into a gradient smeared across
  the boundary vertices.

  We do not have to negotiate with the model. The silhouette is already known exactly
  — it is the alpha of the still we fed Kling — and the geometry is frozen for the
  whole clip by construction, so ONE mask is valid for every frame. Compositing is
  deterministic and costs no generations.

WHAT IT DOES
  1. mask   = hero render pixels darker than --bg-thresh (background is pure 255)
  2. clip   -> frames, mask resized to the frame size
  3. outside the mask -> pure white; inside -> untouched
  4. optional --feather to soften the 1px stair-step so the composite does not
     introduce an edge the texture fit would then try to reproduce

DRIFT CHECK (the reason to run this even on clips that look clean)
  Kling silently pushes in on roughly one clip in three. Per frame we report IoU
  between the mask and the clip's own non-white region. IoU collapsing over the clip
  means the camera or the object moved, and NO amount of masking saves it — the clip
  has to be regenerated. `verdict` is PASS/DRIFT so this is a gate, not a report.

Usage:
  python clamp_to_silhouette.py --video hand_thor.mp4 --render PNG/hand.png \
                                --out out/hand_thor
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--video', help='Kling .mp4 (or use --frames)')
ap.add_argument('--frames', help='directory of already-extracted frames')
ap.add_argument('--render', required=True, help='the still given to Kling, e.g. PNG/hand.png')
ap.add_argument('--out', required=True)
ap.add_argument('--bg-thresh', type=int, default=245,
                help='render pixels with all channels >= this are background. The '
                     'shaded object never exceeds ~194, so the margin is wide.')
ap.add_argument('--feather', type=float, default=1.0, help='mask blur radius in px; 0 = hard edge')
ap.add_argument('--iou-min', type=float, default=0.92,
                help='per-frame IoU below this counts as drift')
ap.add_argument('--drift-frac', type=float, default=0.05,
                help='fraction of frames allowed under --iou-min before the clip fails')
ap.add_argument('--keep-frames', action='store_true')
A = ap.parse_args()

out = Path(A.out); (out / 'frames').mkdir(parents=True, exist_ok=True)


def frame_paths():
    if A.frames:
        return sorted(Path(A.frames).glob('*.png')) or sorted(Path(A.frames).glob('*.jpg'))
    if not A.video:
        sys.exit('need --video or --frames')
    raw = out / '_raw'; raw.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', A.video,
                    str(raw / '%04d.png')], check=True)
    return sorted(raw.glob('*.png'))


def main():
    ren = np.array(Image.open(A.render).convert('RGB'))
    m0 = (ren < A.bg_thresh).any(-1)
    if not m0.any():
        sys.exit(f'{A.render}: no object found — is --bg-thresh right?')

    fps = frame_paths()
    if not fps:
        sys.exit('no frames')
    H, W = np.array(Image.open(fps[0]).convert('RGB')).shape[:2]

    # mask is authored at render resolution; NEAREST keeps the boundary crisp so the
    # threshold below cannot invent a halo of half-object pixels.
    mask = np.array(Image.fromarray((m0 * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127
    if A.feather > 0:
        from PIL import ImageFilter
        alpha = np.array(Image.fromarray((mask * 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(A.feather))).astype(np.float32) / 255.0
    else:
        alpha = mask.astype(np.float32)
    a3 = alpha[..., None]

    rows, spill_px = [], 0
    for i, p in enumerate(fps, 1):
        im = np.array(Image.open(p).convert('RGB')).astype(np.float32)
        occ = (im < A.bg_thresh).any(-1)              # what the clip actually painted

        inter = float((occ & mask).sum()); union = float((occ | mask).sum())
        iou = inter / union if union else 0.0
        outside = int((occ & ~mask).sum()); spill_px += outside

        comp = im * a3 + 255.0 * (1.0 - a3)
        Image.fromarray(comp.clip(0, 255).astype(np.uint8)).save(out / 'frames' / f'{i:04d}.png')
        rows.append(dict(frame=i, iou=iou, outside_px=outside,
                         outside_pct=100.0 * outside / (H * W)))

    ious = np.array([r['iou'] for r in rows])
    bad = int((ious < A.iou_min).sum())
    drift = bad > A.drift_frac * len(rows)
    # a monotone IoU slide is a push-in; a noisy one is just effect spill, which we fixed
    trend = float(np.polyfit(np.arange(len(ious)), ious, 1)[0]) * len(ious)

    rep = dict(video=A.video or A.frames, render=A.render, n=len(rows), size=[W, H],
               iou_mean=float(ious.mean()), iou_min=float(ious.min()),
               iou_trend_over_clip=trend, frames_below_thresh=bad,
               spill_removed_pct=100.0 * spill_px / (len(rows) * H * W),
               verdict='DRIFT' if drift else 'PASS', per_frame=rows)
    json.dump(rep, open(out / 'clamp_report.json', 'w'), indent=1)

    if not A.keep_frames and (out / '_raw').exists():
        for f in (out / '_raw').glob('*.png'):
            f.unlink()
        (out / '_raw').rmdir()

    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30',
                    '-i', str(out / 'frames' / '%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16',
                    str(out / 'clamped.mp4')], check=True)

    print(f"[{Path(A.out).name}] n={len(rows)}  IoU mean={ious.mean():.4f} min={ious.min():.4f} "
          f"trend={trend:+.4f}  spill removed={rep['spill_removed_pct']:.3f}% of frame  "
          f"-> {rep['verdict']}")
    if drift:
        print(f"  {bad}/{len(rows)} frames under IoU {A.iou_min} — object or camera moved, "
              f"REGENERATE. Masking cannot fix geometry drift.")
    print(f"  {out/'clamped.mp4'}")


if __name__ == '__main__':
    main()
