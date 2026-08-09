"""
render_rung14_t2_video.py — the video for the TRELLIS.2 cross-attention arm

rung14_crossattn_lora.py trained to 23.544 dB but only ever wrote per-epoch PNG
strips. This renders the 150-frame comparison:  GT | frozen | rung14.

It reuses the training module rather than reimplementing it — the flow ODE, the
LoRA context, the SurfaceRenderer, the camera and the conditioning all come from
there, so what is rendered is what was trained. Two things are inherited from
run_baseline_field.py's findings and matter here:

  * render from the PBR FIELD, not a UV bake. Baking 430,910 faces into a 2048
    atlas gives 9.7 texels/face and aliases the texture into speckled mud.
    grid_sample_3d at surface points has no such ceiling.
  * encode the shape latent ONCE. It is a deterministic function of the fixed
    mesh, and recomputing it per frame was most of the old 21 s/frame.

Both arms (frozen and adapted) are rendered in the same loop from the same noise
and the same conditioning, so the only difference between the panels is the
1,228,800-parameter adapter on cross_attn.to_q/to_kv/to_out.
"""

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/trellis2_baseline/'
            'runs/rung14_xattn_cross_r4_s42_fb9c291e')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(_RUN))
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--tag', default='rung14_t2')
ap.add_argument('--fps', type=int, default=15)
ARGS = ap.parse_args()

RUN_DIR = Path(ARGS.run)
assert RUN_DIR.exists(), f'no such run: {RUN_DIR}'
CFG = json.load(open(RUN_DIR / 'config.json'))

# The training module parses argv and redirects stdout at import time. Hand it the
# config this run was trained with (anything else builds a registry the checkpoint
# will not fit), then take stdout back.
_real_stdout, _real_stderr = sys.stdout, sys.stderr
sys.argv = ['rung14_crossattn_lora.py',
            '--mesh', '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply',
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
import rung14_crossattn_lora as R          # noqa: E402
sys.stdout, sys.stderr = _real_stdout, _real_stderr

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import trimesh                              # noqa: E402
from PIL import Image, ImageDraw            # noqa: E402

DEVICE = R.DEVICE
OUT = (_HERE / 'out' / ARGS.tag).resolve()
(OUT / 'renders').mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log(f'RUNG14 (TRELLIS.2) VIDEO — GT | frozen | cross-attn LoRA')
    log(f'  run  {RUN_DIR.name}')
    log(f'  ckpt {ARGS.ckpt}   frames 1..{ARGS.n_frames}   res {CFG["resolution"]}')
    log('=' * 88)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{CFG["resolution"]}']
    dec = pipe.models['tex_slat_decoder']
    for m in pipe.models.values():
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)

    mesh_in = trimesh.load('/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/'
                           'frozen_f0075.ply', process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE)
    rend = R.SurfaceRenderer(v_raw, v_pp, faces, 518, CFG['resolution'])
    log(f'\n[MESH] {len(mesh_in.vertices):,} verts   silhouette {rend.n_px:,} px')

    gt_dir = Path(R.args.gt_dir)
    g75 = np.array(Image.open(gt_dir / 'frame_0075.png').convert('RGB')
                   .resize((518, 518), Image.LANCZOS))
    gm = torch.from_numpy(g75.min(axis=2) < 242).to(DEVICE)
    iou = float((rend.mask & gm).sum() / max(int((rend.mask | gm).sum()), 1))
    log(f'[GATE-cam] silhouette IoU vs GT f75 = {iou:.4f}')
    assert iou > 0.80, f'GATE-cam FAILED ({iou:.4f})'
    log('[GATE-cam] PASSED')

    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio)).to(DEVICE)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    sd = st['reg'] if 'reg' in st else st['registry_state']
    reg.load_state_dict(sd)
    reg.eval()
    log(f'[LORA] epoch={st.get("epoch")}  psnr={st.get("psnr", float("nan")):.3f}  '
        f'{sum(p.numel() for p in reg.parameters()):,} params')

    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)
    log(f'[SHAPE] {tuple(shape_slat.feats.shape)} — computed ONCE')

    # Same fixed noise the run trained with, so the two panels differ only by the
    # adapter, not by the draw.
    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))

    frames = list(range(1, ARGS.n_frames + 1))
    raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB'))
            for f in frames]
    alphas = [R.largest_component(r.min(axis=2) < 245) for r in raws]
    u = np.zeros_like(alphas[0])
    for a in alphas:
        u |= a
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))
    log(f'[COND] fixed union crop {bbox}\n')

    def decode_render(x0):
        return rend.sample(dec(x0 * tex_std + tex_mean) * 0.5 + 0.5)

    strip_dir = OUT / 'strip'; strip_dir.mkdir(exist_ok=True)
    LAB = 28
    for k, fi in enumerate(frames, 1):
        r, a = raws[k - 1], alphas[k - 1]
        rgba = np.concatenate([r, (a * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        cond_img = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

        t1 = time.time()
        with torch.no_grad():
            cond = pipe.get_cond([cond_img], CFG['resolution'])['cond']
            x_fz = R.run_ode(flow, noise, cond, ss_n, R.T_PAIRS)
            img_fz = decode_render(x_fz).detach()
            with R.lora_ctx(flow, reg):
                x_lo = R.run_ode(flow, noise, cond, ss_n, R.T_PAIRS)
            img_lo = decode_render(x_lo).detach()
        dt = time.time() - t1

        a_fz = (img_fz[0].cpu().numpy() * 255).astype(np.uint8)
        a_lo = (img_lo[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(a_lo).save(OUT / 'renders' / f'frame_{fi:04d}.png')

        gt_img = np.array(Image.open(gt_dir / f'frame_{fi:04d}.png').convert('RGB')
                          .resize((518, 518), Image.LANCZOS))
        H, W = 518, 518
        canv = Image.fromarray(np.full((H + LAB, W * 3, 3), 18, np.uint8))
        d = ImageDraw.Draw(canv)
        for j, (im, lb) in enumerate([(gt_img, 'GT video'),
                                      (a_fz, 'TRELLIS.2 frozen'),
                                      (a_lo, 'rung14 cross-attn LoRA')]):
            canv.paste(Image.fromarray(im), (j * W, LAB))
            d.rectangle([j * W, 0, (j + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
            d.text((j * W + 8, 8), lb, fill=(240, 240, 240))
        canv.save(strip_dir / f'{k:04d}.png')

        del x_fz, x_lo, img_fz, img_lo
        if k % 10 == 0 or k == 1:
            log(f'  {k:3d}/{len(frames)}  f{fi:04d}  {dt:.1f}s')
        torch.cuda.empty_cache()

    vid = OUT / f'RUNG14_T2_gt_vs_frozen_vs_xattn.mp4'
    pr = subprocess.run(['/usr/bin/ffmpeg', '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(ARGS.fps),
                    '-i', str(strip_dir / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)],
                   check=True)
    log(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)\n[DONE]')


if __name__ == '__main__':
    main()
