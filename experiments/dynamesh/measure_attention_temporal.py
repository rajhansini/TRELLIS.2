"""
measure_attention_temporal.py
=============================
Does the cross-attention CORRESPONDENCE carry the temporal signal, or does only
the value stream change while the attention stays put?

THE QUESTION, PRECISELY
  Across the 150 frames the mesh is fixed, the voxels are fixed, and the noise is
  fixed. The ONLY input that changes is the DINO token set. So for each voxel the
  attention row p_f in R^1029 is a function of frame f alone, and

      how much p_f moves from frame to frame IS the question.

  If p_f barely moves while the rendered texture changes a lot, then cross
  attention is not where the temporal information travels — the model is
  re-weighting a nearly fixed correspondence and the change is carried by the
  VALUES v, not the weights. That is a claim about mechanism, and it decides
  whether editing attention (Fuse3D-style, or our LoRA) can control the dynamics
  at all.

WHAT IS MEASURED, PER FRAME, PER ARM
  Along the REAL 12-step sampling trajectory — the latent is evolving, not held
  at the initial noise, which is what an earlier probe of ours got wrong:

    entropy      H(p)/log(P)      over the 1024 PATCH tokens
    cls_mass     probability on the 5 non-spatial tokens (CLS + 4 registers)
    top1         mass in the strongest 1% of patches
    attn_flick   mean_i || p_f(i) - p_{f-1}(i) ||_1     consecutive-frame change
    attn_drift   mean_i || p_f(i) - p_1(i)     ||_1     change from frame 1

  attn_flick / attn_drift is deliberately the same shape of statistic we already
  report for the rendered texture, so the two are directly comparable: if the
  texture drifts and the attention does not, that gap is the result.

  Everything is also split visible / invisible, because a correspondence that
  only moves on the camera-facing side is a different finding from one that
  moves everywhere.

COST AND WHY IT IS NOT STORED RAW
  150 frames x 12 steps x 1999 voxels x 1029 tokens x 4 B = 14.8 TB. The full
  map is therefore reduced on the fly: per-frame statistics are accumulated, only
  the per-frame statistics are accumulated for the summary table. The FULL
  [voxels x tokens] matrix is written for EVERY frame: pooled over the 12 steps
  and stored fp16 it is 4.1 MB per frame per arm, 1.23 GB for the whole study,
  so the per-voxel / per-token structure is preserved on disk and the CSV is a
  convenience summary rather than the data.

ATTENTION IS RECOMPUTED, NOT READ
  TRELLIS.2 runs cross-attention through a fused kernel that never materialises
  the weights. They are rebuilt from to_q / to_kv with the same reshapes and the
  same RMS norms the real forward applies (attention/modules.py:128-139;
  qk_rms_norm_cross is true in the shipped config). Omitting the norms yields a
  plausible but wrong map.
"""

import argparse, json, math, os, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/trellis2_baseline/'
            'runs/rung14_xattn_cross_r4_s42_fb9c291e')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(_RUN))
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--arms', nargs='+', default=['frozen', 'rung14'])
ap.add_argument('--n-registers', type=int, default=4)
ap.add_argument('--keep-frames', type=int, nargs='+', default=None,
                help='frames whose FULL [voxel x token] map is written; default ALL')
ap.add_argument('--no-maps', action='store_true',
                help='skip the per-frame matrices (summaries only)')
ap.add_argument('--steps-report', type=int, nargs='+', default=None,
                help='ODE step indices to report separately (default: all 12 pooled)')
ap.add_argument('--tag', default='attn_temporal')
ARGS = ap.parse_args()

RUN_DIR = Path(ARGS.run)
CFG = json.load(open(RUN_DIR / 'config.json'))

_so, _se = sys.stdout, sys.stderr
sys.argv = ['rung14_crossattn_lora.py',
            '--mesh', '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply',
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
import rung14_crossattn_lora as R          # noqa: E402
sys.stdout, sys.stderr = _so, _se

import numpy as np                          # noqa: E402
import torch                                # noqa: E402
import trimesh                              # noqa: E402
import nvdiffrast.torch as dr               # noqa: E402
from PIL import Image                       # noqa: E402

DEVICE = R.DEVICE
OUT = (_HERE / 'out' / ARGS.tag).resolve()
(OUT / 'maps').mkdir(parents=True, exist_ok=True)
RES = CFG['resolution']
LAT = RES // 16
GRID = RES // 16
N_SPECIAL = 1 + ARGS.n_registers
P = GRID * GRID


def log(*a):
    print(*a, flush=True)


class TrajectoryAttention:
    """
    Accumulate the mean attention map over blocks and heads, for ONE flow
    evaluation. Reset between ODE steps so each step is measured separately;
    the caller pools them.
    """

    def __init__(self, flow):
        self.flow, self.handles = flow, []
        self.reset()

    def reset(self):
        self.full = None      # [N, M] mean over blocks and heads
        self.n = 0

    def _acc(self, blk, q_out, kv_out):
        ca = blk.cross_attn
        H = ca.num_heads
        q = q_out.reshape(q_out.shape[0], H, -1)
        kv = kv_out.reshape(*kv_out.shape[:2], 2, H, -1)
        k = kv[:, :, 0]
        if getattr(ca, 'qk_rms_norm', False):
            q = ca.q_rms_norm(q)
            k = ca.k_rms_norm(k)
        q, k = q.float(), k[0].float()
        p = torch.softmax(torch.einsum('nhd,mhd->hnm', q, k) / math.sqrt(q.shape[-1]),
                          dim=-1).mean(0)                      # [N, M]
        self.full = p if self.full is None else self.full + p
        self.n += 1
        del p, q, k, kv

    def __enter__(self):
        store = {}

        def mk_q(blk):
            def _h(m, i, o):
                store[id(blk)] = o.feats if hasattr(o, 'feats') else o
            return _h

        def mk_kv(blk):
            def _h(m, i, o):
                qo = store.pop(id(blk), None)
                if qo is not None:
                    with torch.no_grad():
                        self._acc(blk, qo, o)
            return _h

        for blk in self.flow.blocks:
            self.handles.append(blk.cross_attn.to_q.register_forward_hook(mk_q(blk)))
            self.handles.append(blk.cross_attn.to_kv.register_forward_hook(mk_kv(blk)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self):
        return self.full / max(self.n, 1)


def summarise(full, vis):
    """entropy / cls mass / top-1% over the PATCH tokens, split by visibility."""
    spec = full[:, :N_SPECIAL].sum(-1)
    pat = full[:, N_SPECIAL:]
    pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-12)
    ent = -(pat.clamp_min(1e-12).log() * pat).sum(-1) / math.log(P)
    top = pat.topk(max(1, P // 100), dim=-1).values.sum(-1)
    out = {}
    for nm, m in (('all', torch.ones_like(vis)), ('vis', vis), ('inv', ~vis)):
        out[nm] = dict(entropy=float(ent[m].mean()), cls=float(spec[m].mean()),
                       top1=float(top[m].mean()))
    return out, pat


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 96)
    log('TEMPORAL ATTENTION — does the correspondence move with the texture?')
    log(f'  rung14 v2 : {RUN_DIR.name}')
    log(f'  {ARGS.n_frames} frames x 12 ODE steps x 30 blocks, arms={ARGS.arms}')
    log('=' * 96)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{RES}']
    for m in pipe.models.values():
        if isinstance(m, torch.nn.Module):
            for p_ in m.parameters():
                p_.requires_grad_(False)

    mesh_in = trimesh.load('/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/'
                           'frozen_f0075.ply', process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE).contiguous()

    shape_slat = pipe.encode_shape_slat(mesh_pp, RES)
    coords = shape_slat.coords[:, 1:].long()
    N = coords.shape[0]
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std

    # visibility from the training camera (geometry, independent of attention)
    full_proj = (R.intrinsics_to_projection(R.INTRINSICS.to(DEVICE), 0.5, 3.0)
                 @ R.EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full_proj.transpose(-1, -2)).contiguous()
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, faces, (1024, 1024))
    tri = (rast[0, ..., 3].long() - 1); tri = tri[tri >= 0].unique()
    vis_vert = torch.zeros(v_raw.shape[0], dtype=torch.bool, device=DEVICE)
    vis_vert[faces.long()[tri].reshape(-1)] = True
    vidx = ((v_pp + 0.5) * LAT).long().clamp(0, LAT - 1)
    key_v = (vidx[:, 0] * LAT + vidx[:, 1]) * LAT + vidx[:, 2]
    key_c = (coords[:, 0] * LAT + coords[:, 1]) * LAT + coords[:, 2]
    visible = torch.isin(key_c, torch.unique(key_v[vis_vert]))
    log(f'\n[GEOMETRY] {N} voxels, {int(visible.sum())} visible '
        f'({100*float(visible.float().mean()):.1f}%)')

    # conditioning for every frame, fixed union crop (identical framing throughout)
    gt_dir = Path(R.args.gt_dir)
    frames = list(range(1, ARGS.n_frames + 1))
    raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB'))
            for f in frames]
    als = [R.largest_component(r.min(axis=2) < 245) for r in raws]
    u = np.zeros_like(als[0])
    for a in als:
        u |= a
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))
    log(f'[COND] fixed union crop {bbox}; only the DINO tokens vary across frames')

    def cond_of(i):
        rgba = np.concatenate([raws[i], (als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        im = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))
        with torch.no_grad():
            return pipe.get_cond([im], RES)['cond']

    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio)).to(DEVICE)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    reg.eval()
    log(f'[LORA] epoch={st.get("epoch")}  psnr={st.get("psnr", float("nan")):.3f}  '
        f'{sum(p_.numel() for p_ in reg.parameters()):,} params')

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))

    cap = TrajectoryAttention(flow)
    csv = open(OUT / 'attn_temporal.csv', 'w')
    csv.write('arm,frame,entropy_all,cls_all,top1_all,entropy_vis,cls_vis,top1_vis,'
              'entropy_inv,cls_inv,top1_inv,attn_flicker,attn_drift,'
              'attn_flicker_vis,attn_flicker_inv,seconds\n')

    for arm in ARGS.arms:
        log(f'\n{"="*96}\n[{arm.upper()}]  full 12-step trajectory, attention captured '
            f'at every step\n{"="*96}')
        prev = first = None
        with cap:
            for i, fr in enumerate(frames):
                t0 = time.time()
                cond = cond_of(i)
                # walk the REAL trajectory; the latent evolves step to step
                x = noise
                acc = None
                with torch.no_grad():
                    ctxm = R.lora_ctx(flow, reg) if arm == 'rung14' else None
                    if ctxm is not None:
                        ctxm.__enter__()
                    try:
                        for (t, t_prev) in R.T_PAIRS:
                            cap.reset()
                            v = R.flow_eval(flow, x, t, cond, ss_n)
                            m = cap.get()
                            acc = m if acc is None else acc + m
                            x = x - (t - t_prev) * v
                    finally:
                        if ctxm is not None:
                            ctxm.__exit__(None, None, None)
                full = acc / len(R.T_PAIRS)          # pooled over the trajectory
                s, pat = summarise(full, visible)

                if prev is None:
                    fl = fl_v = fl_i = 0.0
                    dr_ = 0.0
                    first = pat.clone()
                else:
                    d = (pat - prev).abs().sum(-1)
                    fl = float(d.mean()); fl_v = float(d[visible].mean())
                    fl_i = float(d[~visible].mean())
                    dr_ = float((pat - first).abs().sum(-1).mean())
                prev = pat.clone()

                dt = time.time() - t0
                csv.write(f'{arm},{fr},{s["all"]["entropy"]:.6f},{s["all"]["cls"]:.6f},'
                          f'{s["all"]["top1"]:.6f},{s["vis"]["entropy"]:.6f},'
                          f'{s["vis"]["cls"]:.6f},{s["vis"]["top1"]:.6f},'
                          f'{s["inv"]["entropy"]:.6f},{s["inv"]["cls"]:.6f},'
                          f'{s["inv"]["top1"]:.6f},{fl:.6e},{dr_:.6e},'
                          f'{fl_v:.6e},{fl_i:.6e},{dt:.2f}\n')
                csv.flush()
                # EVERY frame's full [voxels x tokens] matrix is kept. Pooled over
                # the 12 ODE steps and stored fp16 this is 1999 x 1029 x 2 B =
                # 4.1 MB per frame per arm, 1.23 GB for the whole study — so
                # there is no reason to reduce it and lose the per-voxel,
                # per-token structure the analysis actually needs.
                if not ARGS.no_maps and (ARGS.keep_frames is None
                                         or fr in ARGS.keep_frames):
                    np.save(OUT / 'maps' / f'{arm}_f{fr:04d}.npy',
                            full.cpu().numpy().astype(np.float16))
                if i % 10 == 0 or i == len(frames) - 1:
                    log(f'  {i+1:3d}/{len(frames)}  f{fr:04d}  ent={s["all"]["entropy"]:.4f}'
                        f'  cls={s["all"]["cls"]:.4f}  flick={fl:.3e}  drift={dr_:.3e}'
                        f'  {dt:.1f}s')
                del cond, full, acc
                torch.cuda.empty_cache()
    csv.close()

    np.savez(OUT / 'meta.npz', visible=visible.cpu().numpy(),
             coords=coords.cpu().numpy(), bbox=np.array(bbox))
    log(f'\n[SAVE] {OUT}/attn_temporal.csv   maps for frames {ARGS.keep_frames}')
    log('[DONE]')


if __name__ == '__main__':
    main()
