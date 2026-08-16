"""
render_rung27_orbit.py — turntable for the rung27 SELF-ATTENTION arm

Copy of render_rung17_orbit.py. ONE substantive change: it imports
rung27_selfattn_lora instead of rung17_backproj_lora, because only that module's
CrossAttnLoRA carries sa_qkv / sa_out. Rendering a rung27 checkpoint through the
rung17 trainer raises "Unexpected key(s) in state_dict" -- which is the good
case; the bad case would be a silent partial load.

Views, matching every earlier run so the videos are directly comparable:
    side  --turns 0   camera FIXED at the training view, frames 1..150
    360   --turns 1   one revolution across the sequence
    720   --turns 2   two revolutions, so every surface point is seen TWICE at
                      different points in the texture evolution

--- original header follows ---

render_rung14_t2_orbit.py — the 360 turntable for the TRELLIS.2 cross-attention arm

THE POINT. Frame is PINNED and the camera orbits, so shape and texture are both
constant and anything that changes across the video is viewpoint alone. That is
the only way to see whether the edit reached surface the training camera never
saw. rung13 measured the failure mode on the decoder arm: one camera supervises
16.8% of vertices, the adapter edits 100% of them at equal strength (ratio 1.02),
and R^2 of the edit against the camera's image axis is 0.16-0.23 versus ~0.00 for
the frozen colour. It learned a projection, so it falls apart under rotation.
rung14 moves the adapter upstream of 30 frozen self-attention blocks and bets
that TRELLIS.2's own propagation carries the edit around the object.

EFFICIENCY. The frame is pinned, so the texture latent does not change with the
camera: the ODE runs ONCE per arm and the resulting PBR field is re-sampled at
every angle. 120 angles therefore cost 120 rasterisations, not 120 flow
integrations.

GATE-cam asserts that orbit(yaw=0, elev=0, r=2) reproduces the confirmed
front-view EXTRINSICS, so "the texture is a sticker" can never be an artefact of
a wrong camera.
"""

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/'
            'runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(_RUN))
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--mesh', default=None,
                help="which mesh to render. Defaults to the mesh recorded in the "
                     "run's config.json; falls back to the teapot for runs made "
                     "before the mesh was hashed. Rendering a checkpoint against "
                     "the WRONG mesh produces a plausible video of the wrong "
                     "object with every gate passing, so this is not optional "
                     "once more than one object exists.")
ap.add_argument('--sweep', default='angle', choices=['angle', 'both'],
                help="angle = frame pinned, camera orbits (propagation at one "
                     "instant); both = frame advances 1..N AND camera turns 360 "
                     "together (propagation across the whole sequence)")
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--pin-frame', type=int, default=75)
ap.add_argument('--n-angles', type=int, default=120)
ap.add_argument('--elev', type=float, default=15.0)
ap.add_argument('--radius', type=float, default=2.0)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--frozen-only', action='store_true',
                help='render PURE TRELLIS.2 only — no adapter, single panel')
ap.add_argument('--turns', type=float, default=1.0,
                help='how many full revolutions the camera makes across the '
                     'sequence. 1 = one 360 over the N frames; 2 = 720, i.e. '
                     'every surface point is seen TWICE at different points in '
                     'the texture evolution, which is what separates a texture '
                     'that lives on the surface from one painted on at a fixed '
                     'angle.')
ap.add_argument('--tag', default='rung27_orbit')
ap.add_argument('--fps', type=int, default=20)
ARGS = ap.parse_args()

RUN_DIR = Path(ARGS.run)
CFG = json.load(open(RUN_DIR / 'config.json'))
_TEAPOT = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply'
MESH = ARGS.mesh or CFG.get('mesh') or _TEAPOT
# The CONDITIONING images must come from this run's own data. Without this the
# imported trainer falls back to its --gt-dir default (the teapot's lava frames)
# and every render is this mesh wearing the TEAPOT's texture — plausible-looking
# and completely wrong. Cost six renders on 2026-08-11.
_TEAPOT_GT = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/'
              'teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
GT_DIR = CFG.get('gt_dir') or _TEAPOT_GT
print(f'[MESH]   {MESH}', flush=True)
print(f'[GT-DIR] {GT_DIR}   <- conditioning images', flush=True)

# IMPORTING THE TRAINER CREATES A RUN DIRECTORY. rung*_lora.py runs its argparse
# and mkdirs OUT at MODULE SCOPE, before main(), so a bare import lands an empty
# runs/<label>/ with a zero-byte train.log. That is where the 21 empty
# rung17_all_kv_r{1,2,8,16,32}_* shells came from — orbit renders of the rung22
# rank sweep, whose fake argv omitted --lambda-con so the label fell back to 17.
# Pointing --out-dir at a scratch path stops this file adding to that pile.
_SCRATCH = Path('/tmp') / f'orbit_import_{os.getpid()}'
_so, _se = sys.stdout, sys.stderr
sys.argv = ['rung27_selfattn_lora.py',
            '--mesh', MESH,
            '--gt-dir', GT_DIR,
            '--out-dir', str(_SCRATCH),
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
# rung27, NOT rung17. rung17_backproj_lora.CrossAttnLoRA has no sa_qkv/sa_out, so
# load_state_dict on a rung27 checkpoint raises "Unexpected key(s)". Importing the
# matching trainer is what makes the registry shape line up with the checkpoint.
import rung27_selfattn_lora as R           # noqa: E402
sys.stdout, sys.stderr = _so, _se

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import trimesh                              # noqa: E402
import nvdiffrast.torch as dr               # noqa: E402
from PIL import Image, ImageDraw            # noqa: E402
from flex_gemm.ops.grid_sample import grid_sample_3d   # noqa: E402

DEVICE = R.DEVICE
OUT = (_HERE / 'out' / ARGS.tag).resolve()
(OUT / 'frames').mkdir(parents=True, exist_ok=True)
NEAR, FAR = 0.5, 3.0


def log(*a):
    print(*a, flush=True)


def orbit_extrinsics(yaw_deg, elev_deg, radius):
    """Camera on a sphere looking at the origin, in v1's MeshRenderer convention."""
    y, e = math.radians(yaw_deg), math.radians(elev_deg)
    eye = np.array([radius * math.cos(e) * math.sin(y),
                    -radius * math.cos(e) * math.cos(y),
                    radius * math.sin(e)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ext = np.eye(4)
    ext[0, :3], ext[1, :3], ext[2, :3] = right, -up, fwd
    ext[:3, 3] = -ext[:3, :3] @ eye
    return torch.tensor(ext, dtype=torch.float32)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log('RUNG27 (TRELLIS.2) TURNTABLE — frozen | cross-attn + SELF-ATTN LoRA')
    log(f'  frame {ARGS.pin_frame} PINNED, {ARGS.n_angles} angles, elev {ARGS.elev}')
    log('=' * 88)

    # GATE-cam before anything expensive
    d = float((orbit_extrinsics(0.0, 0.0, 2.0) - R.EXTRINSICS).abs().max())
    log(f'[GATE-cam] |orbit(0,0,2) - confirmed EXTRINSICS| = {d:.3e}')
    assert d < 1e-5, f'GATE-cam FAILED ({d:.3e}): orbit camera != confirmed front view'
    log('[GATE-cam] PASSED')

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

    mesh_in = trimesh.load(MESH, process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE).contiguous()

    # alpha MUST come from the run's own config, not the default: dW is scaled by
    # alpha/rank, so rendering a rank-32 checkpoint with the wrong alpha silently
    # rescales every edit. Runs made before the rank sweep have no 'lora_alpha'
    # key; 4.0 is their implied value and at their rank 4 it gives scaling 1.0,
    # which is exactly how they were trained.
    _alpha = CFG.get('lora_alpha', 4.0)
    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio),
                         targets=tuple(CFG['target_set']),
                         active=CFG.get('active'), alpha=_alpha).to(DEVICE)
    print(f'[LORA-SCALE] rank {CFG["rank"]}  alpha {_alpha}  '
          f'-> scaling {_alpha / CFG["rank"]:.4f}', flush=True)
    print(f'[REGISTRY] targets={tuple(CFG["target_set"])}  '
          f'blocks={len(CFG.get("active", range(30)))}  '
          f'{sum(p.numel() for p in reg.parameters()):,} params', flush=True)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    reg.eval()
    log(f'[LORA] epoch={st.get("epoch")}  psnr={st.get("psnr", float("nan")):.3f}  '
        f'{sum(p.numel() for p in reg.parameters()):,} params on cross_attn')

    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)

    # Conditioning, preprocessed exactly as training did. The crop box is the
    # UNION over every frame used, so framing is identical frame to frame and
    # contributes nothing to what the video shows.
    gt_dir = Path(R.args.gt_dir)
    _fr_list = ([ARGS.pin_frame] if ARGS.sweep == 'angle'
                else list(range(1, ARGS.n_frames + 1)))
    _raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB'))
             for f in _fr_list]
    _als = [R.largest_component(r.min(axis=2) < 245) for r in _raws]
    _u = np.zeros_like(_als[0])
    for a in _als:
        _u |= a
    ys, xs = np.where(_u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    def cond_image(i):
        rgba = np.concatenate([_raws[i], (_als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        return Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    cond_img = cond_image(0)

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))

    def decode_both(ci):
        """Run the ODE for the frozen and adapted arms from the SAME noise and
        the SAME conditioning, so the only difference is the adapter."""
        with torch.no_grad():
            c = pipe.get_cond([ci], CFG['resolution'])['cond']
            a = dec(R.run_ode(flow, noise, c, ss_n, R.T_PAIRS) * tex_std + tex_mean) * 0.5 + 0.5
            if ARGS.frozen_only:
                return a, None          # no adapter run at all
            with R.lora_ctx(flow, reg):
                xl = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            b = dec(xl * tex_std + tex_mean) * 0.5 + 0.5
        return a, b


    pbr_fz = pbr_lo = None
    if ARGS.sweep == 'angle':
        # frame pinned -> the field never changes -> integrate ONCE, then the
        # angles cost rasterisation only.
        t0 = time.time()
        pbr_fz, pbr_lo = decode_both(cond_img)
        log(f'[FIELD] both arms decoded once in {time.time()-t0:.1f}s — '
            f'{ARGS.n_angles} angles now cost rasterisation only')

    ctx = dr.RasterizeCudaContext()
    intr = R.INTRINSICS.to(DEVICE)
    proj = R.intrinsics_to_projection(intr, NEAR, FAR)
    res = ARGS.res
    grid_res = CFG['resolution']

    def render_at(ext, pbr):
        full = (proj @ ext.to(DEVICE)).unsqueeze(0)
        vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, faces, (res, res))
        m = rast[0, ..., 3] > 0
        idx = torch.nonzero(m.reshape(-1)).squeeze(1)
        pos = dr.interpolate(v_pp.unsqueeze(0).contiguous(), rast, faces)[0]
        pos = pos[0].reshape(-1, 3)[idx].contiguous()
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((pos + 0.5) * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        rgb = attrs[..., :3].reshape(-1, 3).float().clamp(0, 1)
        base = torch.ones(res * res, 3, device=DEVICE)
        img = base.index_put((idx,), rgb).view(res, res, 3)
        return (img.cpu().numpy() * 255).astype(np.uint8)

    _rg = CFG.get('rung', '?')
    _tg = '+'.join(t.replace('to_', '') for t in CFG.get('target_set', []))
    _LORA_LABEL = f'rung{_rg} LoRA  ({_tg})'
    LAB = 28
    if ARGS.sweep == 'angle':
        items = [(0, ARGS.pin_frame, i * 360.0 / ARGS.n_angles)
                 for i in range(ARGS.n_angles)]
    else:
        n = ARGS.n_frames
        items = [(i, i + 1, i * 360.0 * ARGS.turns / n) for i in range(n)]
    yaws = items
    for k, (ci, fr, yaw) in enumerate(items, 1):
        ext = orbit_extrinsics(yaw, ARGS.elev, ARGS.radius)
        if ARGS.sweep == 'both':
            pbr_fz, pbr_lo = decode_both(cond_image(ci))
        with torch.no_grad():
            a_fz = render_at(ext, pbr_fz)
            a_lo = None if ARGS.frozen_only else render_at(ext, pbr_lo)
        panels = ([(a_fz, 'TRELLIS.2 (pure, no adapter)')] if ARGS.frozen_only
                  else [(a_fz, 'TRELLIS.2 frozen'), (a_lo, _LORA_LABEL)])
        canv = Image.fromarray(np.full((res + LAB, res * len(panels), 3), 18, np.uint8))
        d_ = ImageDraw.Draw(canv)
        for j, (im, lb) in enumerate(panels):
            canv.paste(Image.fromarray(im), (j * res, LAB))
            d_.rectangle([j * res, 0, (j + 1) * res - 1, LAB - 1], fill=(38, 38, 58))
            _t = (f'{lb}   yaw {yaw:5.1f}' if ARGS.sweep == 'angle'
                  else f'{lb}   frame {fr:3d}   yaw {yaw:5.1f}')
            d_.text((j * res + 8, 8), _t, fill=(240, 240, 240))
        canv.save(OUT / 'frames' / f'{k:04d}.png')
        if ARGS.sweep == 'both':
            del pbr_fz, pbr_lo
            torch.cuda.empty_cache()
        if k % 20 == 0 or k == 1:
            log(f'  {k:3d}/{len(items)}  frame={fr}  yaw={yaw:5.1f}')

    # Name the file after WHAT IT IS. This used to be a constant, so every arm
    # produced RUNG14_T2_BOTH_frozen_vs_xattn.mp4 and scp-ing three of them into
    # one directory silently left you with only the last.
    # turns=0 is the FIXED training view, not '0x360' -- name it for what it is
    _turns = ('sideview' if ARGS.turns == 0
              else f'{ARGS.turns:g}x360'.replace('1x360', '360'))
    # tau belongs in the name too: two rung20 arms differ ONLY by tau, and
    # without it both write the same filename and clobber each other on scp.
    _cw = CFG.get('conf_weight'); _ct = CFG.get('conf_tau')
    _tau = f'_tau{_ct:g}' if _cw else ''
    vid = OUT / f'rung{_rg}_{_tg}{_tau}_{_turns}_{ARGS.sweep}_vs_frozen.mp4'
    pr = subprocess.run(['/usr/bin/ffmpeg', '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(ARGS.fps),
                    '-i', str(OUT / 'frames' / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)], check=True)
    log(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)\n[DONE]')


if __name__ == '__main__':
    main()
