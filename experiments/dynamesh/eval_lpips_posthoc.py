"""
eval_lpips_posthoc.py — add the missing LPIPS column to finished runs.

WHY THIS EXISTS
  rung27's final_eval.json reports PSNR and SSIM only. A w_lpips sweep that never
  reports LPIPS cannot answer the reviewer question it was run to answer: it tunes
  a perceptual weight and shows no perceptual number. This computes LPIPS (plus
  PSNR/SSIM recomputed the same way, so all three come from one pass) from renders
  already on disk. rung27_selfattn_lora.py is NOT modified.

INPUT — the fixed-view render, not an orbit
  render_rung27_orbit.py --turns 0 puts the camera at the TRAINING view and
  advances frames 1..N ("side" in that script's own docstring). LPIPS has to be
  measured on the same view PSNR/SSIM were, so --turns 0 is the only valid source;
  a 360/720 orbit would score views the loss never saw.

  Those frames are composites: [label strip] over N panels of `res` each. The
  adapted render is the LAST panel. --panel selects a different one.

TARGETS
  gt_targets_<asset>/frames/gt_%04d.png — the rendered backprojected GT the loss
  itself used, NOT the raw video frame. Same mesh, same camera, so the silhouette
  matches exactly and the comparison is texture-only.

Usage:
  python eval_lpips_posthoc.py \
      --render-dir out/render_r27_horse_w0.1/frames \
      --gt-dir     out/gt_targets_horse_metal_guan/frames \
      --out        out/lpips_eval_w0.1.json
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault('HF_HOME', '/net/scratch/rajhansini/.cache/huggingface')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--render-dir', required=True, help='frames/ from --turns 0 render')
ap.add_argument('--gt-dir', required=True, help='gt_targets_<asset>/frames')
ap.add_argument('--out', required=True)
ap.add_argument('--n-frames', type=int, default=121)
ap.add_argument('--panel', type=int, default=-1,
                help='which panel holds the adapted render; -1 = last')
ap.add_argument('--label-h', type=int, default=28, help='height of the label strip')
ap.add_argument('--tag', default='')
args = ap.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R, G = Path(args.render_dir), Path(args.gt_dir)


def load_panel(p: Path):
    """Composite -> the adapted panel, as float RGB in [0,1]."""
    im = np.array(Image.open(p).convert('RGB'))
    h, w = im.shape[:2]
    body = im[args.label_h:]                      # drop the label strip
    res = body.shape[0]
    n = max(1, w // res)
    j = (n - 1) if args.panel < 0 else args.panel
    return body[:, j * res:(j + 1) * res].astype(np.float32) / 255.0, n


def psnr(a, b, m):
    d = ((a - b) ** 2)[m]
    mse = float(d.mean())
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 100.0


def main():
    import lpips as lpips_pkg
    from skimage.metrics import structural_similarity as ssim_fn
    net = lpips_pkg.LPIPS(net='alex', verbose=False).to(DEVICE).eval()

    rows, npan = [], None
    for i in range(1, args.n_frames + 1):
        rp, gp = R / f'{i:04d}.png', G / f'gt_{i:04d}.png'
        if not rp.exists() or not gp.exists():
            print(f'  skip {i}: missing {"render" if not rp.exists() else "gt"}', flush=True)
            continue
        pred, npan = load_panel(rp)
        gt = np.array(Image.open(gp).convert('RGB')).astype(np.float32) / 255.0
        if gt.shape != pred.shape:
            gt = (np.array(Image.fromarray((gt * 255).astype(np.uint8))
                           .resize((pred.shape[1], pred.shape[0]), Image.LANCZOS))
                  .astype(np.float32) / 255.0)

        # score only where the object is; background is white in both and would
        # inflate PSNR toward infinity while telling us nothing about texture.
        mask = (gt < 0.99).any(-1)
        m3 = np.repeat(mask[..., None], 3, -1)

        with torch.no_grad():
            t = lambda x: (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
                           .permute(2, 0, 1)[None] * 2 - 1).to(DEVICE)
            lp = float(net(t(pred), t(gt)).item())
        rows.append(dict(frame=i,
                         psnr=psnr(pred, gt, m3),
                         ssim=float(ssim_fn(pred, gt, channel_axis=2, data_range=1.0)),
                         lpips=lp))
        if i % 30 == 0:
            print(f'  {i}/{args.n_frames}', flush=True)

    if not rows:
        sys.exit('no frame pairs found — check --render-dir / --gt-dir')

    agg = {k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
    agg |= {k + '_std': float(np.std([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
    out = dict(tag=args.tag, n=len(rows), panels_detected=npan,
               render_dir=str(R), gt_dir=str(G), **agg, per_frame=rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(f"\n[{args.tag or 'eval'}] n={len(rows)}  PSNR={agg['psnr']:.3f}  "
          f"SSIM={agg['ssim']:.4f}  LPIPS={agg['lpips']:.4f}  -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
