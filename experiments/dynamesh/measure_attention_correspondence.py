"""
measure_attention_correspondence.py
===================================
Do the BACK voxels attend to anything meaningful, or do they smear attention
over the whole image?

WHY THE OBVIOUS VERSION OF THIS QUESTION IS ILL-POSED
  Cross-attention softmaxes over tokens, so EVERY voxel's attention sums to 1.
  "Does it attend" is trivially yes. The question that has content is whether the
  distribution is CONCENTRATED (the voxel reads a specific image region -> a real
  correspondence) or DIFFUSE (the voxel receives an average of the whole image ->
  no correspondence, just a global colour prior).

  So the quantities are:
      entropy      H(p)/log(M)      0 = one patch, 1 = uniform over all patches
      top-1% mass  how much probability sits in the strongest 1% of patches
      centroid     sum_j p_j * (x_j, y_j)   where the attention lands in the image

THE PART THAT MAKES IT A MEASUREMENT AND NOT A STORY
  For VISIBLE voxels we know the answer independently: project the voxel into the
  conditioning image with the confirmed camera and that is where its attention
  SHOULD land. So we can validate the instrument on the voxels that have ground
  truth, report the projection error in patch units, and only then interpret what
  the invisible voxels are doing. Without that step "the back attends diffusely"
  would be unfalsifiable.

  Reported split four ways: {frozen, rung14} x {visible, invisible}.

TOKEN LAYOUT
  DINOv3 ViT-L/16, image_size 512, num_register_tokens 4:
      token 0        CLS
      tokens 1..4    registers
      tokens 5..1028 the 32x32 patch grid, row-major
  1 + 4 + 1024 = 1029, which is exactly the conditioning length. CLS and
  registers are EXCLUDED from the spatial statistics — they carry no location,
  and leaving them in would inflate concentration for free. Their mass is
  reported separately, because a voxel dumping its attention on CLS is precisely
  a voxel with no spatial correspondence.

ATTENTION IS RECOMPUTED, NOT READ
  TRELLIS.2 runs cross-attention through a fused kernel that never materialises
  the weights. They are rebuilt here from to_q / to_kv with the same reshapes and
  the same RMS norms the real forward applies (attention/modules.py:128-139,
  qk_rms_norm_cross=true in the shipped config). Skipping the norms gives a
  plausible but wrong map.
"""

import argparse, json, math, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/trellis2_baseline/'
            'runs/rung14_xattn_cross_r4_s42_fb9c291e')

ap = argparse.ArgumentParser()
ap.add_argument('--run', default=str(_RUN))
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--frame', type=int, default=75)
ap.add_argument('--timestep', type=float, default=1.0,
                help='where in the ODE to probe; correspondence is sharpest early')
ap.add_argument('--n-registers', type=int, default=4)
ap.add_argument('--tag', default='attn_corr')
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
LAT = RES // 16
GRID = RES // 16          # DINOv3 patch grid side at image_size=RES (patch 16)
N_SPECIAL = 1 + ARGS.n_registers


def log(*a):
    print(*a, flush=True)


class PatchAttention:
    """
    Accumulate, per voxel, the mean attention distribution over PATCH tokens.

    Kept as [N, GRID*GRID] averaged over blocks and heads, plus a per-head copy,
    plus the mass landing on CLS/register tokens. The full [H, N, M] map is freed
    per block, so 30 blocks cost one block's memory.
    """

    def __init__(self, flow):
        self.flow = flow
        self.handles = []
        self.reset()

    def reset(self):
        self.patch = None       # [N, P]  head-averaged
        self.per_head = None    # [H, N, P]
        self.special = None     # [N]  mass on CLS + registers
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
                          dim=-1)                                  # [H, N, M]
        spec = p[..., :N_SPECIAL].sum(-1).mean(0)                  # [N]
        pat = p[..., N_SPECIAL:]                                   # [H, N, P]
        self.special = spec if self.special is None else self.special + spec
        self.per_head = pat if self.per_head is None else self.per_head + pat
        self.patch = pat.mean(0) if self.patch is None else self.patch + pat.mean(0)
        self.n += 1
        del p, pat, q, k, kv

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

    def result(self):
        n = max(self.n, 1)
        pat = self.patch / n
        pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-12)   # renormalise
        return pat, (self.per_head / n), (self.special / n)


def stats(pat, grid):
    """entropy (normalised), top-1% mass, centroid in patch units."""
    P = pat.shape[-1]
    ent = -(pat.clamp_min(1e-12).log() * pat).sum(-1) / math.log(P)
    k = max(1, P // 100)
    top = pat.topk(k, dim=-1).values.sum(-1)
    ys, xs = torch.meshgrid(torch.arange(grid, device=pat.device, dtype=torch.float32),
                            torch.arange(grid, device=pat.device, dtype=torch.float32),
                            indexing='ij')
    cx = (pat * xs.reshape(-1)).sum(-1)
    cy = (pat * ys.reshape(-1)).sum(-1)
    return ent, top, torch.stack([cx, cy], -1)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 92)
    log('ATTENTION CORRESPONDENCE — do back voxels read a REGION, or the whole image?')
    log(f'  frame {ARGS.frame}   t={ARGS.timestep}   patch grid {GRID}x{GRID}   '
        f'{N_SPECIAL} special tokens excluded')
    log('=' * 92)

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
    coords = shape_slat.coords[:, 1:].long()
    N = coords.shape[0]
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std

    # ── conditioning, and the crop that maps GT pixels -> patch grid ─────────
    gt_dir = Path(R.args.gt_dir)
    raw = np.array(Image.open(gt_dir / f'frame_{ARGS.frame:04d}.png').convert('RGB'))
    Hgt, Wgt = raw.shape[:2]
    al = R.largest_component(raw.min(axis=2) < 245)
    ys_, xs_ = np.where(al)
    cx0, cy0 = (xs_.min() + xs_.max()) / 2, (ys_.min() + ys_.max()) / 2
    sz = int(max(xs_.max() - xs_.min(), ys_.max() - ys_.min()))
    bbox = (int(cx0 - sz // 2), int(cy0 - sz // 2), int(cx0 + sz // 2), int(cy0 + sz // 2))
    rgba = np.concatenate([raw, (al * 255).astype(np.uint8)[..., None]], -1)
    fcr = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
    cond_img = Image.fromarray(((fcr[:, :, :3] * fcr[:, :, 3:4]) * 255).astype(np.uint8))
    with torch.no_grad():
        cond = pipe.get_cond([cond_img], RES)['cond']
    assert cond.shape[1] == N_SPECIAL + GRID * GRID, (
        f'token count {cond.shape[1]} != {N_SPECIAL} special + {GRID*GRID} patches; '
        f'the layout assumption is wrong and every centroid below would be garbage')
    log(f'\n[COND] tokens {tuple(cond.shape)} = {N_SPECIAL} special + {GRID*GRID} patches'
        f'   crop {bbox}')

    # ── ground truth: where does each voxel PROJECT in the conditioning image ─
    # The mesh is registered to the video (silhouette IoU 0.9058), so projecting
    # a voxel's vertices with the confirmed camera gives the image location its
    # attention SHOULD favour — for the voxels the camera can see.
    full = (R.intrinsics_to_projection(R.INTRINSICS.to(DEVICE), 0.5, 3.0)
            @ R.EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2))[0]
    ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
    px = (ndc[:, 0] * 0.5 + 0.5) * Wgt                      # GT-frame pixels
    py = (0.5 - ndc[:, 1] * 0.5) * Hgt
    # GT pixel -> crop -> patch grid
    qx = (px - bbox[0]) / max(bbox[2] - bbox[0], 1) * GRID
    qy = (py - bbox[1]) / max(bbox[3] - bbox[1], 1) * GRID

    vidx = ((v_pp + 0.5) * LAT).long().clamp(0, LAT - 1)
    key_v = (vidx[:, 0] * LAT + vidx[:, 1]) * LAT + vidx[:, 2]
    key_c = (coords[:, 0] * LAT + coords[:, 1]) * LAT + coords[:, 2]
    order = torch.argsort(key_c)
    pos_in_c = torch.searchsorted(key_c[order], key_v)
    pos_in_c = pos_in_c.clamp(max=N - 1)
    match = key_c[order][pos_in_c] == key_v
    vox_of_vert = torch.where(match, order[pos_in_c], torch.full_like(pos_in_c, -1))

    proj = torch.zeros(N, 2, device=DEVICE)
    cnt = torch.zeros(N, device=DEVICE)
    good = vox_of_vert >= 0
    proj.index_add_(0, vox_of_vert[good], torch.stack([qx, qy], -1)[good])
    cnt.index_add_(0, vox_of_vert[good], torch.ones(int(good.sum()), device=DEVICE))
    has_proj = cnt > 0
    proj = proj / cnt.clamp_min(1).unsqueeze(-1)

    # visibility, by rasterising and taking the triangles actually hit
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, torch.bmm(vh, full.transpose(-1, -2)).contiguous(),
                           faces, (1024, 1024))
    tri = (rast[0, ..., 3].long() - 1)
    tri = tri[tri >= 0].unique()
    vis_vert = torch.zeros(v_raw.shape[0], dtype=torch.bool, device=DEVICE)
    vis_vert[faces.long()[tri].reshape(-1)] = True
    visible = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    visible[vox_of_vert[good & vis_vert]] = True
    log(f'[GEOMETRY] {N} voxels; {int(visible.sum())} visible '
        f'({100*float(visible.float().mean()):.1f}%), {int(has_proj.sum())} with a projection')

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

    cap = PatchAttention(flow)
    out = {}
    for arm in ('frozen', 'rung14'):
        cap.reset()
        with torch.no_grad(), cap:
            if arm == 'frozen':
                R.flow_eval(flow, noise, ARGS.timestep, cond, ss_n)
            else:
                with R.lora_ctx(flow, reg):
                    R.flow_eval(flow, noise, ARGS.timestep, cond, ss_n)
        pat, per_head, spec = cap.result()
        ent, top, cen = stats(pat, GRID)
        out[arm] = dict(pat=pat, per_head=per_head, spec=spec, ent=ent, top=top, cen=cen)
        log(f'  [{arm}] {cap.n} blocks captured')
    cap.__exit__()

    uni_ent = 1.0                       # normalised: uniform == 1
    uni_top = 0.01                      # top 1% of a uniform distribution
    log(f'\n{"="*92}')
    log('PER-VOXEL ATTENTION OVER THE 32x32 PATCH GRID')
    log(f'  reference: uniform gives entropy {uni_ent:.3f}, top-1% mass {uni_top:.3f}')
    log(f'{"arm":9}{"set":11}{"n":>6}{"entropy":>10}{"top1%":>9}'
        f'{"CLS+reg":>10}{"proj err":>11}')
    log('-' * 92)
    rows = {}
    for arm in ('frozen', 'rung14'):
        d = out[arm]
        err = (d['cen'] - proj).norm(dim=-1)
        for name, m in (('visible', visible & has_proj),
                        ('invisible', (~visible) & has_proj)):
            if int(m.sum()) == 0:
                continue
            r = dict(n=int(m.sum()),
                     entropy=float(d['ent'][m].mean()),
                     top1=float(d['top'][m].mean()),
                     special=float(d['spec'][m].mean()),
                     proj_err=float(err[m].mean()),
                     proj_err_med=float(err[m].median()))
            rows[f'{arm}/{name}'] = r
            log(f'{arm:9}{name:11}{r["n"]:>6}{r["entropy"]:>10.4f}{r["top1"]:>9.4f}'
                f'{r["special"]:>10.4f}{r["proj_err"]:>11.2f}')
    log('-' * 92)
    log('proj err is in PATCH units (grid is 32 wide); it is only meaningful for the')
    log('visible set, where the projection is ground truth. A small value there says')
    log('the attention map IS a 2D-3D correspondence, which is what licenses reading')
    log('the invisible row at all. Chance level for a random centroid is ~12 patches.')
    log('=' * 92)

    # per-head, on invisible voxels — is the far side carried by a few heads?
    log('\nPER-HEAD entropy on INVISIBLE voxels (lower = that head reads a region, '
        'not the whole image)')
    for arm in ('frozen', 'rung14'):
        ph = out[arm]['per_head']
        ph = ph / ph.sum(-1, keepdim=True).clamp_min(1e-12)
        e = -(ph.clamp_min(1e-12).log() * ph).sum(-1) / math.log(ph.shape[-1])
        m = (~visible) & has_proj
        log(f'  {arm:8} ' + '  '.join(f'h{h}={float(e[h][m].mean()):.3f}'
                                      for h in range(e.shape[0])))

    np.savez(OUT / 'attn_corr.npz',
             visible=visible.cpu().numpy(), has_proj=has_proj.cpu().numpy(),
             proj=proj.cpu().numpy(), coords=coords.cpu().numpy(),
             **{f'{a}_{k}': out[a][k].cpu().numpy()
                for a in out for k in ('pat', 'ent', 'top', 'cen', 'spec')})
    json.dump(dict(frame=ARGS.frame, timestep=ARGS.timestep, n_voxels=int(N),
                   grid=GRID, n_special=N_SPECIAL,
                   visible=int(visible.sum()), rows=rows),
              open(OUT / 'attn_corr.json', 'w'), indent=2)
    log(f'\n[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
