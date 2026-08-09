"""
measure_alignment.py — which voxels does the image actually control?

THE QUESTION
  rung14 puts a LoRA on the SLat-flow cross-attention betting that TRELLIS.2's
  own 3D propagation carries the edit to surface the training camera never saw.
  The turntable says it partly does; "partly" is not a number.

THE INSTRUMENT (after Fuse3D, arXiv 2602.17040)
  TRELLIS's cross-attention IS a 2D<->3D correspondence: attn[voxel, token] says
  which image patch each voxel reads from. Normalising it two ways answers two
  different questions:

    FORWARD  softmax over the TOKEN axis   -> for this voxel, which region does
                                              it read?  (peakiness = how
                                              decisively it reads anything)
    REVERSE  softmax over the VOXEL axis   -> for this token, which voxels does
                                              it claim?  By complement: which
                                              voxels NO token claims.

  That complement is the orphaned set. The hypothesis this script tests is that
  the orphaned set is essentially the surface the camera cannot see — and that
  if rung14 propagates, the adapter SHRINKS it.

WHAT IS MEASURED
  1. per-voxel forward peak and reverse claim, averaged over blocks, per head
  2. per-voxel VISIBILITY from the training camera, by rasterising the mesh and
     mapping visible vertices to their latent voxel
  3. the cross-tabulation of the two: is orphaned == unseen?
  4. all of the above for the FROZEN model and for rung14, from identical noise
     and conditioning, so the only difference is the adapter

  Nothing is trained. This runs on the existing checkpoint.

REPLICATING THE ATTENTION EXACTLY
  TRELLIS.2 computes cross-attention through a fused kernel that never
  materialises the weights, so they are recomputed here from to_q / to_kv. The
  1024-config sets qk_rms_norm_cross=true, so q_rms_norm and k_rms_norm ARE
  applied before the dot product (attention/modules.py:128-139). Skipping them
  would produce a plausible but wrong map.
"""

import argparse, json, math, os, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/trellis2_baseline/'
            'runs/rung14_xattn_cross_r4_s42_fb9c291e')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(_RUN))
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--frame', type=int, default=75)
ap.add_argument('--timesteps', type=float, nargs='+', default=[1.0, 0.75, 0.5, 0.25])
ap.add_argument('--tag', default='alignment')
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
OUT.mkdir(parents=True, exist_ok=True)
RES = CFG['resolution']
LAT = RES // 16          # shape encoder downsamples 16x; latent grid side


def log(*a):
    print(*a, flush=True)


class AttnCapture:
    """
    Recompute the cross-attention weights TRELLIS.2's fused kernel throws away.

    Hooks to_q and to_kv, rebuilds q and k with the same reshapes and the same
    RMS norms the real forward applies, and accumulates per-voxel statistics.
    Only [H, N] summaries are kept — the full [H, N, M] map is freed per block,
    so 30 blocks cost one block's memory, not thirty.
    """

    def __init__(self, flow):
        self.flow = flow
        self.handles = []
        self.reset()

    def reset(self):
        self.fwd = None     # [H, N] mean over blocks of  max_m softmax_m(logits)
        self.rev = None     # [H, N] mean over blocks of  max_m softmax_n(logits)
        self.n_blocks = 0

    def _accumulate(self, blk, q_out, kv_out):
        ca = blk.cross_attn
        H = ca.num_heads
        q = q_out.reshape(q_out.shape[0], H, -1)                    # [N, H, hd]
        kv = kv_out.reshape(*kv_out.shape[:2], 2, H, -1)            # [B, M, 2, H, hd]
        k = kv[:, :, 0]                                             # [B, M, H, hd]
        if getattr(ca, 'qk_rms_norm', False):
            q = ca.q_rms_norm(q)
            k = ca.k_rms_norm(k)
        q = q.float()
        k = k[0].float()                                            # [M, H, hd]
        hd = q.shape[-1]
        logits = torch.einsum('nhd,mhd->hnm', q, k) / math.sqrt(hd)  # [H, N, M]
        fwd = torch.softmax(logits, dim=-1).amax(dim=-1)             # [H, N]
        rev = torch.softmax(logits, dim=-2).amax(dim=-1)             # [H, N]
        self.fwd = fwd if self.fwd is None else self.fwd + fwd
        self.rev = rev if self.rev is None else self.rev + rev
        self.n_blocks += 1
        del logits, q, k, kv

    def __enter__(self):
        store = {}

        def mk_q(blk):
            def _h(mod, inp, out):
                store[id(blk)] = out.feats if hasattr(out, 'feats') else out
            return _h

        def mk_kv(blk):
            def _h(mod, inp, out):
                qo = store.pop(id(blk), None)
                if qo is not None:
                    with torch.no_grad():
                        self._accumulate(blk, qo, out)
            return _h

        for blk in self.flow.blocks:
            self.handles.append(blk.cross_attn.to_q.register_forward_hook(mk_q(blk)))
            self.handles.append(blk.cross_attn.to_kv.register_forward_hook(mk_kv(blk)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def result(self):
        n = max(self.n_blocks, 1)
        return (self.fwd / n).cpu().numpy(), (self.rev / n).cpu().numpy()


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log('ORPHANED-VOXEL MEASUREMENT — is the unaligned set the unseen surface?')
    log(f'  frame {ARGS.frame}   timesteps {ARGS.timesteps}   latent grid {LAT}^3')
    log('=' * 88)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{RES}']
    for m in pipe.models.values():
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)

    mesh_in = trimesh.load('/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/'
                           'frozen_f0075.ply', process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE).contiguous()

    shape_slat = pipe.encode_shape_slat(mesh_pp, RES)
    coords = shape_slat.coords[:, 1:].long()                 # [N, 3] latent voxel idx
    N = coords.shape[0]
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    log(f'\n[LATENT] {N} voxels, coord range '
        f'{coords.min(0).values.tolist()}..{coords.max(0).values.tolist()}')

    # ── per-voxel VISIBILITY from the training camera ────────────────────────
    # Rasterise, take the triangles actually hit, collect their vertices, and map
    # each to the latent voxel it falls in. A voxel counts as seen if any of its
    # vertices is seen. This is the ground truth the alignment set is tested
    # against — it is geometry, not attention.
    ctx = dr.RasterizeCudaContext()
    full = (R.intrinsics_to_projection(R.INTRINSICS.to(DEVICE), 0.5, 3.0)
            @ R.EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    rast, _ = dr.rasterize(ctx, clip, faces, (1024, 1024))
    tri = (rast[0, ..., 3].long() - 1)
    tri = tri[tri >= 0].unique()
    vis_vert = torch.zeros(v_raw.shape[0], dtype=torch.bool, device=DEVICE)
    vis_vert[faces.long()[tri].reshape(-1)] = True
    vidx = ((v_pp + 0.5) * LAT).long().clamp(0, LAT - 1)      # vertex -> latent voxel
    key_v = (vidx[:, 0] * LAT + vidx[:, 1]) * LAT + vidx[:, 2]
    key_c = (coords[:, 0] * LAT + coords[:, 1]) * LAT + coords[:, 2]
    seen_keys = torch.unique(key_v[vis_vert])
    visible = torch.isin(key_c, seen_keys)
    log(f'[VISIBILITY] {int(visible.sum())}/{N} voxels seen from the training '
        f'camera = {100*float(visible.float().mean()):.1f}%   '
        f'(rung13 measured 16.8% of VERTICES)')

    # ── conditioning, exactly as training built it ───────────────────────────
    gt_dir = Path(R.args.gt_dir)
    raw = np.array(Image.open(gt_dir / f'frame_{ARGS.frame:04d}.png').convert('RGB'))
    al = R.largest_component(raw.min(axis=2) < 245)
    ys, xs = np.where(al)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))
    rgba = np.concatenate([raw, (al * 255).astype(np.uint8)[..., None]], -1)
    fcr = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
    cond_img = Image.fromarray(((fcr[:, :, :3] * fcr[:, :, 3:4]) * 255).astype(np.uint8))
    with torch.no_grad():
        cond = pipe.get_cond([cond_img], RES)['cond']
    log(f'[COND] tokens {tuple(cond.shape)}')

    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio)).to(DEVICE)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    reg.eval()
    log(f'[LORA] epoch={st.get("epoch")}  psnr={st.get("psnr", float("nan")):.3f}')

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))

    cap = AttnCapture(flow)
    results = {}
    for arm in ('frozen', 'rung14'):
        per_t = {}
        for t in ARGS.timesteps:
            cap.reset()
            with torch.no_grad(), cap:
                if arm == 'frozen':
                    R.flow_eval(flow, noise, t, cond, ss_n)
                else:
                    with R.lora_ctx(flow, reg):
                        R.flow_eval(flow, noise, t, cond, ss_n)
            fwd, rev = cap.result()
            per_t[str(t)] = dict(fwd=fwd, rev=rev)
            log(f'  [{arm}] t={t}: {cap.n_blocks} blocks captured')
        results[arm] = per_t
    cap.__exit__()

    vis = visible.cpu().numpy()
    uniform = 1.0 / N
    log(f'\n{"="*88}\nORPHANED VOXELS   (reverse claim below {3}x uniform = {3*uniform:.2e})')
    log(f'{"arm":10}{"t":>6}{"orphaned":>12}{"of unseen":>12}{"of seen":>10}'
        f'{"claim seen":>12}{"claim unseen":>14}')
    log('-' * 88)
    summary = {}
    for arm in ('frozen', 'rung14'):
        for t in ARGS.timesteps:
            rev = results[arm][str(t)]['rev'].mean(axis=0)        # mean over heads
            orph = rev < 3 * uniform
            row = dict(
                orphaned=int(orph.sum()),
                orphaned_frac=float(orph.mean()),
                frac_of_unseen_orphaned=float(orph[~vis].mean()) if (~vis).any() else 0.0,
                frac_of_seen_orphaned=float(orph[vis].mean()) if vis.any() else 0.0,
                mean_claim_seen=float(rev[vis].mean()) if vis.any() else 0.0,
                mean_claim_unseen=float(rev[~vis].mean()) if (~vis).any() else 0.0)
            summary[f'{arm}@{t}'] = row
            log(f'{arm:10}{t:>6.2f}{row["orphaned"]:>12}'
                f'{row["frac_of_unseen_orphaned"]:>12.3f}{row["frac_of_seen_orphaned"]:>10.3f}'
                f'{row["mean_claim_seen"]:>12.3e}{row["mean_claim_unseen"]:>14.3e}')
    log('=' * 88)

    # per-head sharpness — Fuse3D found only some heads carry the correspondence
    log('\nPER-HEAD reverse claim on UNSEEN voxels (higher = that head reaches '
        'the far side); Fuse3D used heads 1,5,13 on TRELLIS v1')
    for arm in ('frozen', 'rung14'):
        rev = results[arm][str(ARGS.timesteps[0])]['rev']          # [H, N]
        per_head = rev[:, ~vis].mean(axis=1) if (~vis).any() else rev.mean(axis=1)
        log(f'  {arm:8} ' + '  '.join(f'h{h}={v:.2e}' for h, v in enumerate(per_head)))

    np.savez(OUT / 'alignment.npz', visible=vis, coords=coords.cpu().numpy(),
             **{f'{a}_{t}_{k}': results[a][str(t)][k]
                for a in results for t in ARGS.timesteps for k in ('fwd', 'rev')})
    json.dump(dict(frame=ARGS.frame, n_voxels=int(N),
                   visible_voxels=int(vis.sum()),
                   visible_frac=float(vis.mean()),
                   uniform=uniform, summary=summary),
              open(OUT / 'alignment.json', 'w'), indent=2)
    log(f'\n[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
