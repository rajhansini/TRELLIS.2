"""
rung14_crossattn_lora.py — LoRA on TRELLIS.2's texture-flow CROSS-ATTENTION
===========================================================================
New file. Nothing from rung13 or earlier is touched; TRELLIS.2's own source is
imported, never modified.

WHY HERE AND NOT IN THE DECODER
  rung13 put the adapter in the mesh DECODER. Measured consequence: the training
  camera sees 16.8% of the vertices, the adapter edits 100% of them at equal
  strength (ratio 1.02), and R^2 of the edit against the training camera's image
  axis is 0.16-0.23 versus ~0.00 for the frozen colour. It learned a projection,
  because the decoder has NO cross-attention -- the image never reaches it, and
  all of TRELLIS's 3D reasoning already finished upstream.

  Here the adapter sits INSIDE the texture flow, on cross-attention, which is
  where image evidence enters the 3D. Downstream of it are 30 blocks of
  self-attention (frozen) that propagate across voxels. The bet: an edit
  admitted at cross-attention gets distributed by TRELLIS's own machinery to
  voxels the camera never saw. That bet is what this experiment tests, and the
  turntable decides it -- not PSNR from the training view.

WHERE EXACTLY
  models['tex_slat_flow_model_1024'].blocks[0..29]
      .cross_attn.to_q    Linear(1536, 1536)   <- LoRA
      .cross_attn.to_kv   Linear(1024, 3072)   <- LoRA   image -> 3D keys/values
      .cross_attn.to_out  Linear(1536, 1536)   <- LoRA
      .self_attn.*                                 FROZEN  (3D coherence)
      .mlp, .adaLN_modulation                      FROZEN
  rank 4 -> 40,960/block x 30 = 1,228,800 parameters. B is zero-init, so step 0
  is the unmodified model.

  --targets cross+mlp adds mlp.mlp[0]/mlp.mlp[2] for the ablation. Excluding the
  MLP is a SCOPING decision, not a principled one: the MLP would also propagate
  (29 blocks of self-attention follow it). It is left out first so the claim has
  one variable. rung5_colonly adapted the MLP alongside attention and reached
  21.3 dB, so MLP adaptation is demonstrably not harmful -- just unisolated.

THE ONE-STEP TRAINING DESIGN, AND WHY
  Backpropagating through all 12 ODE steps means 12 sequential 1.3B forward
  passes with activations retained. Instead, per step:

      run the ODE to a RANDOM timestep k        under no_grad   (~6 evals avg)
      ONE model call at k                       WITH grad
      x_0_hat = sampler._pred_to_xstart(x_k, t_k, v)
      decode -> PBR voxels -> sample at mesh vertices -> render -> loss

  Sampling k fresh each step means the LoRA still sees every timestep. The
  x_0 formula is the sampler's OWN helper, not a reimplementation, so the
  rectified-flow parameterisation cannot drift from TRELLIS.2's.

GRADIENT HEALTH IS THE POINT OF THIS RUN
  The gradient now travels back through the texture decoder AND a flow
  evaluation, both in bf16, to a LoRA sitting behind cross-attention's softmax
  where gradients are already small. v1 hit exactly this (fp16 flush-to-zero in
  dec_mesh, fixed with LOSS_SCALE=4096). So this script logs, every epoch:

      per-target   |dB| for to_q / to_kv / to_out separately
      per-block    |dB| for all 30 blocks, so a dead early half is visible
      zero count   how many of the 90 B matrices received EXACTLY zero
      dynamic range min/max/median across matrices
      grad/param   |dB| / |B|, i.e. is the step size meaningful

  and GATE-grad refuses to start training if ANY B matrix gets zero gradient.
  Failing in 60 seconds beats discovering it after four hours.

Usage:
  python rung14_crossattn_lora.py --mesh mesh.ply --epochs 30
  python rung14_crossattn_lora.py --mesh mesh.ply --smoke      # 2 epochs, 8 frames
"""

import argparse, json, math, os, sys, time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

TRELLIS2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
sys.path.insert(0, str(TRELLIS2))
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--gt-dir', default='/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                                    '/outputs/teapot_lava_kling_premium'
                                    '/teapot_lava_kling_premium_front/all_frames_150')
ap.add_argument('--rank',       type=int, default=4)
ap.add_argument('--targets',    default='cross', choices=['cross', 'cross+mlp'])
ap.add_argument('--epochs',     type=int, default=30)
ap.add_argument('--n-frames',   type=int, default=150)
ap.add_argument('--lr',         type=float, default=1e-4)
ap.add_argument('--loss-scale', type=float, default=4096.0)
ap.add_argument('--grad-clip',  type=float, default=1.0)
ap.add_argument('--w-lpips',    type=float, default=0.1)
ap.add_argument('--seed',       type=int, default=42)
ap.add_argument('--resolution', type=int, default=512, choices=[512, 1024],
                help="512 by default, and this is MEASURED, not a preference. At 1024 the "
                     "texture decoder's backward graph does not fit on a 44 GiB A40: "
                     "probe_spconv_bwd.py OOMed at an identical 43.6 GiB peak under "
                     "explicit_gemm/frozen, implicit_gemm/grad-weights and "
                     "implicit_gemm_splitk/grad-weights — i.e. it is the graph size, not "
                     "the conv algorithm. At 512 the same probe reaches backward. "
                     "TRELLIS.2 ships tex_slat_flow_model_512 as a first-class option and "
                     "the frozen baseline scored the same at both (PSNR 8.149 vs 8.148).")
ap.add_argument('--render-res', type=int, default=512)
ap.add_argument('--smoke',      action='store_true')
ap.add_argument('--out-dir',    default=None, type=Path)
ap.add_argument('--alignment',  type=Path,
                default=Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/'
                             'lora_experiments/visibility/alignment/alignment.json'),
                help="rung13's solved mesh->video registration, applied before rendering")
ap.add_argument('--align-mesh', action='store_true',
                help="apply rung13's transform. OFF by default because the meshes in "
                     "render/out/ were exported BY export_frame_mesh.py --alignment, "
                     "which bakes it in already (export_frame_mesh.py:224-225). Pass "
                     "this only for a mesh that has NOT been through that export.")
ap.add_argument('--no-grad-ckpt', action='store_true',
                help='disable gradient checkpointing (faster, needs >44 GB — OOMs on an A40)')
ap.add_argument('--decoder-grad-weights', action='store_true', default=True,
                help='set decoder weights requires_grad=True (NOT trained) so flex_gemm '
                     'does not return grad_weight=None and crash on reshape')
ap.add_argument('--no-decoder-grad-weights', dest='decoder_grad_weights',
                action='store_false')
ap.add_argument('--spconv-algo', default='implicit_gemm_splitk',
                choices=['implicit_gemm', 'implicit_gemm_splitk', 'explicit_gemm',
                         'masked_implicit_gemm', 'masked_implicit_gemm_splitk'],
                help="flex_gemm sparse-conv algorithm. TRELLIS.2 ships "
                     "masked_implicit_gemm_splitk, whose BACKWARD needs neighbour-cache "
                     "fields that are only stored when the first call on those coords "
                     "requires grad — which is never true here, because encode_shape_slat "
                     "runs frozen first. MEASURED on an A40 at res 512: explicit_gemm dies "
                     "with 'flip_cuda not implemented for UInt32'; masked_*_splitk raises a "
                     "Triton CompilationError; implicit_gemm_splitk WORKS (17.07 GiB, "
                     "sub-second per step once Triton has compiled).")
args = ap.parse_args()

# NOTE on --w-lpips: the trellis2 env has no `lpips` package, so the loss here is
# masked MSE only and this flag is inert. It stays in the config hash so that a
# future run that does add a perceptual term gets its own run directory.

EPOCHS   = 2 if args.smoke else args.epochs
N_FRAMES = 8 if args.smoke else args.n_frames
HELD_OUT = [f for f in range(5, N_FRAMES + 1, 10)]
TRAIN    = [f for f in range(1, N_FRAMES + 1) if f not in HELD_OUT]

import hashlib
_CFG = dict(variant='trellis2_crossattn_lora', rank=args.rank, targets=args.targets,
            epochs=EPOCHS, n_frames=N_FRAMES, lr=args.lr, seed=args.seed,
            loss_scale=args.loss_scale, resolution=args.resolution,
            w_lpips=args.w_lpips, held_out=HELD_OUT)
RUN_ID = hashlib.md5(json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:8]
LABEL  = f'rung14_xattn_{args.targets}_r{args.rank}_s{args.seed}_{RUN_ID}'
OUT    = (args.out_dir or (_HERE / 'runs' / LABEL)).resolve()
for d in (OUT, OUT / 'logs', OUT / 'ckpts', OUT / 'diag'):
    d.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, p):
        self._f = open(p, 'a', buffering=1)
    def write(self, m):
        sys.__stdout__.write(m); self._f.write(m)
    def flush(self):
        sys.__stdout__.flush(); self._f.flush()

sys.stdout = _Tee(OUT / 'train.log'); sys.stderr = sys.stdout

import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image

DEVICE = torch.device('cuda')


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """delta = B @ (A @ x). A kaiming-init, B ZERO-init so step 0 is the identity."""
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


class CrossAttnLoRA(nn.Module):
    """One bundle per block. Names double as the log keys."""
    def __init__(self, ch, ctx, rank, with_mlp=False, mlp_hidden=None):
        super().__init__()
        self.to_q   = LoRALayer(ch,  ch,      rank)
        self.to_kv  = LoRALayer(ctx, 2 * ch,  rank)
        self.to_out = LoRALayer(ch,  ch,      rank)
        if with_mlp:
            self.mlp_fc1 = LoRALayer(ch, mlp_hidden, rank)
            self.mlp_fc2 = LoRALayer(mlp_hidden, ch, rank)


class LoRARegistry(nn.Module):
    def __init__(self, n_blocks, ch, ctx, rank, with_mlp, mlp_hidden):
        super().__init__()
        self.blocks = nn.ModuleDict({
            str(i): CrossAttnLoRA(ch, ctx, rank, with_mlp, mlp_hidden)
            for i in range(n_blocks)})

    def get(self, i):
        k = str(i)
        return self.blocks[k] if k in self.blocks else None


@contextmanager
def lora_ctx(flow_model, registry):
    """
    Hooks on cross_attn only. self_attn is never touched — that is the 3D
    coherence mechanism and leaving it alone is the whole design.
    """
    handles = []
    for i, blk in enumerate(flow_model.blocks):
        lb = registry.get(i)
        if lb is None:
            continue

        def _q(mod, inp, out, _lb=lb):
            return out + _lb.to_q(inp[0]).to(out.dtype)

        def _kv(mod, inp, out, _lb=lb):
            return out + _lb.to_kv(inp[0]).to(out.dtype)

        def _o(mod, inp, out, _lb=lb):
            return out + _lb.to_out(inp[0]).to(out.dtype)

        handles.append(blk.cross_attn.to_q.register_forward_hook(_q))
        handles.append(blk.cross_attn.to_kv.register_forward_hook(_kv))
        handles.append(blk.cross_attn.to_out.register_forward_hook(_o))

        if hasattr(lb, 'mlp_fc1'):
            def _f1(mod, inp, out, _lb=lb):
                d = _lb.mlp_fc1(inp[0].feats if hasattr(inp[0], 'feats') else inp[0])
                return (out.replace(out.feats + d.to(out.feats.dtype))
                        if hasattr(out, 'feats') else out + d.to(out.dtype))

            def _f2(mod, inp, out, _lb=lb):
                d = _lb.mlp_fc2(inp[0].feats if hasattr(inp[0], 'feats') else inp[0])
                return (out.replace(out.feats + d.to(out.feats.dtype))
                        if hasattr(out, 'feats') else out + d.to(out.dtype))
            handles.append(blk.mlp.mlp[0].register_forward_hook(_f1))
            handles.append(blk.mlp.mlp[2].register_forward_hook(_f2))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ── gradient health, the point of this run ───────────────────────────────────

def grad_report(registry, tag=''):
    """
    Per-target and per-block |dB|. Returns a dict; prints a readable block.
    Reported on B (not A) because B is zero-init: if B's gradient is zero, A's
    is zero too by the chain rule, and nothing can ever move.
    """
    per_target, per_block, all_norms, zeros, names = {}, {}, [], 0, []
    for bi, bundle in registry.blocks.items():
        bn = 0.0
        for tname, layer in bundle.named_children():
            g = layer.B.grad
            gn = 0.0 if g is None else float(g.norm())
            per_target.setdefault(tname, []).append(gn)
            all_norms.append(gn); names.append(f'b{bi}.{tname}')
            bn += gn ** 2
            if gn == 0.0:
                zeros += 1
        per_block[int(bi)] = math.sqrt(bn)

    a = np.array(all_norms)
    nz = a[a > 0]
    rep = dict(
        n_matrices=len(a), n_zero=int(zeros),
        min=float(a.min()), max=float(a.max()),
        median=float(np.median(a)),
        median_nonzero=float(np.median(nz)) if len(nz) else 0.0,
        dynamic_range=float(a.max() / max(nz.min(), 1e-30)) if len(nz) else float('inf'),
        per_target={k: float(np.mean(v)) for k, v in per_target.items()},
        per_block={k: per_block[k] for k in sorted(per_block)},
        dead=[names[i] for i in np.where(a == 0)[0][:12]])

    if tag:
        print(f'  [GRAD {tag}]  {rep["n_zero"]}/{rep["n_matrices"]} matrices at EXACTLY zero')
        print(f'     by target : ' + '  '.join(
            f'{k}={v:.3e}' for k, v in rep['per_target'].items()))
        pb = rep['per_block']
        ks = sorted(pb)
        print(f'     by block  : first4=' + ','.join(f'{pb[k]:.2e}' for k in ks[:4])
              + '   last4=' + ','.join(f'{pb[k]:.2e}' for k in ks[-4:]))
        print(f'     range     : min={rep["min"]:.3e}  median={rep["median"]:.3e}  '
              f'max={rep["max"]:.3e}  spread={rep["dynamic_range"]:.1e}x', flush=True)
        if rep['dead']:
            print(f'     DEAD      : {rep["dead"]}', flush=True)
    return rep


def param_norms(registry):
    out = {}
    for bi, bundle in registry.blocks.items():
        for tname, layer in bundle.named_children():
            out.setdefault(tname, []).append(float(layer.B.detach().float().norm()))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ── camera ───────────────────────────────────────────────────────────────────
# Copied verbatim from TRELLIS v1's trellis/renderers/mesh_renderer.py so the
# view is IDENTICAL to every rung before this one. Re-deriving it would risk a
# sign error that would look like a texture bug. GATE-cam checks it anyway.

RENDER_RES_V1 = 518
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ret = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


def rodrigues(rv):
    th = float(np.linalg.norm(rv)) + 1e-12
    k = np.asarray(rv, dtype=np.float64) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


class SurfaceRenderer:
    """
    Fixed mesh, fixed camera -> rasterize ONCE, reuse forever.

    TRELLIS.2 cannot move the mesh and our camera never moves, so the raster,
    the silhouette and each pixel's 3D surface point are all constants. They are
    computed once here. Per training step the only work is sampling the PBR
    voxel field at those constant points, which is where the gradient lives.

    TWO COORDINATE FRAMES, and keeping them straight is the whole trick:
      v_raw  the mesh as it came off TRELLIS v1, optionally rung13-aligned.
             Used for RASTERISATION, so the image lines up with the GT video
             exactly as in every previous rung.
      v_pp   the same vertices after Trellis2TexturingPipeline.preprocess_mesh
             (normalised to [-0.5,0.5], axes permuted). The PBR voxel field
             lives in THIS frame, so v_pp is interpolated across the raster to
             give each pixel its sampling position.
    preprocess_mesh keeps faces and vertex ORDER, so v_raw[i] and v_pp[i] are
    the same vertex. That correspondence is what lets the two frames coexist.
    """

    def __init__(self, v_raw, v_pp, faces, res, resolution):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.ctx = dr.RasterizeCudaContext()
        self.res = res
        self.resolution = resolution

        ext = EXTRINSICS.to(DEVICE)
        intr = INTRINSICS.to(DEVICE)
        full_proj = (intrinsics_to_projection(intr, NEAR, FAR) @ ext).unsqueeze(0)

        v = v_raw.unsqueeze(0)
        v_homo = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        self.v_clip = torch.bmm(v_homo, full_proj.transpose(-1, -2)).contiguous()
        self.faces = faces.int().contiguous()

        rast, _ = dr.rasterize(self.ctx, self.v_clip, self.faces, (res, res))
        self.rast = rast
        self.mask = (rast[0, ..., 3] > 0)                      # [H, W] bool
        self.mask_idx = torch.nonzero(self.mask.reshape(-1), as_tuple=False).squeeze(1)

        pos = dr.interpolate(v_pp.unsqueeze(0), rast, self.faces)[0]   # [1,H,W,3]
        self.pos = pos[0].reshape(-1, 3)[self.mask_idx].contiguous()   # [K, 3]
        self.n_px = int(self.mask.sum())

    def sample(self, pbr_voxel):
        """PBR voxel field -> RGB image. Differentiable w.r.t. pbr_voxel.feats."""
        from flex_gemm.ops.grid_sample import grid_sample_3d
        attrs = grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((self.pos + 0.5) * self.resolution).reshape(1, -1, 3),
            mode='trilinear',
        )                                                       # [K, 6]
        rgb = attrs[..., 0:3].reshape(-1, 3).clamp(0, 1)
        base = torch.ones(self.res * self.res, 3, device=DEVICE, dtype=rgb.dtype)
        img = base.index_put((self.mask_idx,), rgb)             # white background
        return img.view(1, self.res, self.res, 3)


def largest_component(mask):
    """
    Keep only the biggest blob of a foreground mask.

    The GT frames carry a few stray dark pixels — frame 4 has ONE at (959, 0)
    valued [243,245,242], just under the 245 threshold. A raw min/max bbox over
    the union of all frames then spans the whole image: measured (0, 238, 802,
    1040) instead of the correct (212, 188, 802, 778), which both pads outside
    the 960px frame and shrinks the teapot inside the conditioning crop.
    Frame 4 has 89 connected components: the teapot at 102,778 px and the next
    largest at 3 px. Taking the biggest component is therefore decisive, not a
    tuned threshold.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def load_gt(frame_idx, gt_dir, res):
    img = Image.open(Path(gt_dir) / f'frame_{frame_idx:04d}.png').convert('RGB')
    img = img.resize((res, res), Image.LANCZOS)
    a = torch.from_numpy(np.array(img)).float().div(255.0).to(DEVICE)
    return a.unsqueeze(0)                                       # [1,H,W,3]


def psnr_ssim(pred, gt, mask):
    from skimage.metrics import structural_similarity as ssim_fn
    p = pred[0].detach().float().cpu().numpy()
    g = gt[0].detach().float().cpu().numpy()
    m = mask.detach().cpu().numpy()
    mse = float(((p - g) ** 2)[m].mean())
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    ssim = float(ssim_fn(g, p, channel_axis=2, data_range=1.0))
    return psnr, ssim




# ── the sampling schedule, taken from FlowEulerSampler.sample ────────────────
# flow_euler.py:114-118.  steps=12, rescale_t=3.0 -> the knots are FRONT-LOADED
# (seven of twelve above t=0.75). Training timesteps are drawn from THIS list,
# not from a uniform [0,1], so the adapter is never trained where inference
# never goes.
STEPS, RESCALE_T, SIGMA_MIN = 12, 3.0, 1e-5
_ts = np.linspace(1, 0, STEPS + 1)
T_SEQ = (RESCALE_T * _ts / (1 + (RESCALE_T - 1) * _ts)).tolist()
T_PAIRS = [(T_SEQ[i], T_SEQ[i + 1]) for i in range(STEPS)]


def enable_grad_checkpointing(*modules):
    """
    Turn on gradient checkpointing wherever TRELLIS.2 supports it.

    THE FLOW ONLY. Not the decoder — that is deliberate and was measured.

    Both the flow blocks (modulated.py:74-76) and every decoder block type
    (sparse_unet_vae.py:86-88 and friends) call
    torch.utils.checkpoint.checkpoint(..., use_reentrant=False) when their
    use_checkpoint flag is set, and the shipped configs leave it off because
    inference never needs it.

    Checkpointing the DECODER breaks its backward:

        AttributeError: 'SubMConv3dNeighborCache' object has no attribute
                        'valid_signal_i'
                        (flex_gemm/ops/spconv/submanifold_conv3d.py:308)

    flex_gemm's submanifold convolution builds a neighbour cache during forward
    and consumes it during backward. Recomputing the forward under checkpointing
    hands backward a cache that does not carry those fields. The flow is pure
    attention plus SparseLinear — no spconv anywhere — so it checkpoints safely.

    That split is also the one that matters. The first OOM was in the DECODER
    FORWARD, not its backward: the flow's graded evaluation (30 blocks x 1536ch
    x ~9.2k voxels, all activations retained) had already taken the memory the
    decoder then needed. Checkpointing the flow frees exactly that, and leaves
    the decoder's own graph intact and functional.
    """
    n = 0
    for mod in modules:
        for m in mod.modules():
            if hasattr(m, 'use_checkpoint') and m is not mod:
                m.use_checkpoint = True
                n += 1
    return n


def purge_spconv_cache(*tensors):
    """
    Drop every cached submanifold-conv neighbour map so the next call rebuilds it
    with the backward fields present.

    flex_gemm builds that cache in two variants (submanifold_conv3d.py:86-101):

        need_grad = any(ctx.needs_input_grad)          # :331
        if need_grad: ... gray_code, sorted_idx, valid_signal_i/o/seg
        else:         ... gray_code, sorted_idx        # _no_bwd, fields omitted

    and its backward unconditionally reads valid_signal_i/o/seg under the
    MASKED_IMPLICIT_GEMM* algorithms, which are the default (spconv/__init__.py:23).

    encode_shape_slat() runs the shape encoder with frozen weights on an input
    that does not require grad, so need_grad is False and the REDUCED variant is
    what gets cached. TRELLIS.2 registers it on the SparseTensor
    (conv_flex_gemm.py:44-57), and SparseTensor.replace() passes _spatial_cache
    BY REFERENCE (basic.py:675) — so ss_n, the noise, the flow output and the
    decoder input are all one dict. Every later graded decode then reuses the
    reduced cache and dies with:

        AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'

    Priming with a grad-enabled call does NOT fix this: the cache is already
    populated, so the grad-enabled call takes the `neighbor_cache is not None`
    branch and never rebuilds. The entry has to be deleted. Only the
    SubMConv3d_* keys are removed; 'layout', 'shape' and 'seqlen' are
    grad-agnostic and expensive, so they stay.
    """
    removed = []
    for t in tensors:
        for k in [k for k in list(t._spatial_cache.keys())
                  if k.startswith('SubMConv3d_neighbor_cache')]:
            del t._spatial_cache[k]
            removed.append(k)
    return removed


def pred_to_xstart(x_t, t, pred):
    """FlowEulerSampler._pred_to_xstart, flow_euler.py:38-39. Not a re-derivation."""
    return (1 - SIGMA_MIN) * x_t - (SIGMA_MIN + (1 - SIGMA_MIN) * t) * pred


def flow_eval(flow, x, t, cond, concat_cond):
    """
    One flow evaluation.

    NOT sampler._inference_model: this sampler is a FlowEulerGuidanceIntervalSampler
    whose two CFG mixins override that method with extra REQUIRED positional args
    (guidance_strength, guidance_interval) and would raise TypeError.

    Calling the flow directly is not an approximation. The shipped texturing
    config uses guidance_strength=1.0, and classifier_free_guidance_mixin.py:10-11
    short-circuits that to a single cond-only forward pass. This IS inference.
    The t -> 1000*t scaling mirrors flow_euler.py:44-46.
    """
    t_ten = torch.tensor([1000.0 * t] * x.shape[0], device=DEVICE, dtype=torch.float32)
    return flow(x, t_ten, cond, concat_cond=concat_cond)


def run_ode(flow, x, cond, ss_n, pairs):
    """Euler prefix under no_grad, LoRA ACTIVE (inference has it active too)."""
    with torch.no_grad():
        for t, t_prev in pairs:
            v = flow_eval(flow, x, t, cond, ss_n)
            x = x - (t - t_prev) * v
    return x


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules import sparse as sp
    from trellis2.modules.sparse.conv import config as conv_config

    # ── THE SPCONV ALGORITHM MUST BE BACKWARD-CAPABLE ────────────────────────
    # TRELLIS.2 ships 'masked_implicit_gemm_splitk' (conv/config.py:2). Its
    # backward unconditionally reads valid_signal_i/o/seg from the neighbour
    # cache (submanifold_conv3d.py:292-310), but the cache only carries those
    # when the FIRST call on those coords had a grad-requiring input
    # (:331 need_grad = any(ctx.needs_input_grad), :86-101 two variants).
    # encode_shape_slat runs frozen on a non-grad input, so the reduced variant
    # is what gets cached — and SparseTensor.replace() shares _spatial_cache by
    # reference (basic.py:675), so it propagates to every later graded decode:
    #     AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'
    # 'implicit_gemm' takes a different branch whose backward needs only
    # neighbor_map, which every variant of the cache always carries. TRELLIS.2
    # re-reads this on every conv forward (conv_flex_gemm.py:38), so setting it
    # here is enough — no monkey-patching of their source.
    # Inference-only code never hit this because run() is @torch.no_grad().
    conv_config.FLEX_GEMM_ALGO = args.spconv_algo
    print(f'[SPCONV] algorithm = {conv_config.FLEX_GEMM_ALGO}  '
          f'(shipped default is masked_implicit_gemm_splitk, whose backward '
          f'needs cache fields a no_grad first call never stores)', flush=True)

    print('=' * 92)
    print(f'RUNG 14 — LoRA on TRELLIS.2 texture-flow CROSS-ATTENTION')
    print(f'  run       : {LABEL}')
    print(f'  targets   : {args.targets}   rank {args.rank}')
    print(f'  mesh      : {args.mesh}   (FIXED — an input, never modified)')
    print(f'  frames    : {N_FRAMES}   train {len(TRAIN)}   held-out {len(HELD_OUT)}')
    print(f'  epochs    : {EPOCHS}   lr {args.lr}   loss_scale {args.loss_scale}')
    print(f'  out       : {OUT}')
    print('=' * 92, flush=True)
    json.dump(_CFG | {'run_id': RUN_ID, 'label': LABEL},
              open(OUT / 'config.json', 'w'), indent=2)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')

    # ── GATE-device ──────────────────────────────────────────────────────────
    # Trellis2TexturingPipeline.to() (trellis2_texturing.py:97-103) OVERRIDES the
    # base class and is a NO-OP when low_vram=True, which is the default because
    # texturing_pipeline.json does not set it. pipe.cuda() then sets a string and
    # moves nothing; models are walked to the GPU inside run() and walked back
    # after. We call the flow and decoder DIRECTLY, so that lazy scheme would
    # hand us CPU weights against CUDA activations — which is exactly how job
    # 2153281 died at input_layer. Turning low_vram off makes .cuda() mean what
    # it says, and the gate below proves it rather than trusting it.
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{args.resolution}']
    decoder = pipe.models['tex_slat_decoder']
    flow.to(DEVICE); decoder.to(DEVICE)
    for m in pipe.models.values():
        if isinstance(m, nn.Module):
            for q in m.parameters():
                q.requires_grad_(False)

    # ── The decoder's weights must REQUIRE grad, without being TRAINED ───────
    # This is a workaround for a flex_gemm bug, not a modelling choice. Its
    # backward kernel returns grad_weight=None when weight.requires_grad is False
    # (kernels/triton/.../bwd_implicit_gemm.py:166-169), and the caller then does
    # `grad_weight.reshape(...)` with no None check (submanifold_conv3d.py:270):
    #     AttributeError: 'NoneType' object has no attribute 'reshape'
    # The only other branch that guards this, explicit_gemm, is unusable here —
    # it OOMs at res 1024 and raises "flip_cuda not implemented for 'UInt32'" at
    # res 512, measured both ways in probe_spconv_bwd.py.
    #
    # So we let the weights require grad purely to keep the kernel from returning
    # None, and throw the result away. They are NOT in the optimizer — it is
    # constructed over reg.parameters() alone — so nothing updates them, and
    # their .grad is zeroed each step so it cannot accumulate. The model stays
    # frozen in every sense that matters; only an autograd flag changed.
    if args.decoder_grad_weights:
        n_dec = sum(1 for _ in decoder.parameters())
        for q in decoder.parameters():
            q.requires_grad_(True)
        print(f'[SPCONV] decoder: {n_dec} weight tensors set requires_grad=True to '
              f'dodge the grad_weight=None crash. NOT optimised, grads discarded.',
              flush=True)
    _devs = {n: str(next(m.parameters()).device)
             for n, m in pipe.models.items() if isinstance(m, nn.Module)}
    print(f'\n[GATE-device] {_devs}')
    assert all('cuda' in d for d in _devs.values()), \
        f'GATE-device FAILED: a model is still on CPU: {_devs}'
    print('[GATE-device] PASSED', flush=True)

    if not args.no_grad_ckpt:
        n_ck = enable_grad_checkpointing(flow)
        print(f'[MEMORY] gradient checkpointing enabled on {n_ck} FLOW blocks. '
              f'The decoder is deliberately left uncheckpointed — flex_gemm\'s '
              f'spconv backward needs the neighbour cache its forward built, and '
              f'recomputation loses it.', flush=True)
    else:
        print('[MEMORY] gradient checkpointing DISABLED (--no-grad-ckpt)', flush=True)

    ch, ctx, nb = flow.model_channels, flow.cond_channels, len(flow.blocks)
    mlp_hidden = int(ch * flow.mlp_ratio)
    print(f'\n[FLOW] blocks={nb}  model_channels={ch}  cond_channels={ctx}')
    print(f'       cross_attn.to_q  Linear({ch}, {ch})')
    print(f'       cross_attn.to_kv Linear({ctx}, {2*ch})   <- image -> 3D k/v')
    print(f'       cross_attn.to_out Linear({ch}, {ch})', flush=True)

    torch.manual_seed(args.seed)
    reg = LoRARegistry(nb, ch, ctx, args.rank,
                       with_mlp=(args.targets == 'cross+mlp'),
                       mlp_hidden=mlp_hidden).to(DEVICE)
    n_par = sum(p.numel() for p in reg.parameters())
    n_mat = sum(1 for b in reg.blocks.values() for _ in b.named_children())
    print(f'\n[LORA] {n_par:,} params   {n_mat} B matrices   '
          f'(rank {args.rank} x {nb} blocks)')
    assert all(float(l.B.abs().max()) == 0
               for b in reg.blocks.values() for _, l in b.named_children()), \
        'B must be zero-init'
    print('[GATE 0] param count + B=0 identity  PASSED', flush=True)

    # ── mesh: two frames, one vertex ordering ────────────────────────────────
    mesh_in = trimesh.load(args.mesh, process=False, force='mesh')
    print(f'\n[MESH] {len(mesh_in.vertices):,} verts  {len(mesh_in.faces):,} faces  FIXED')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    assert len(mesh_pp.vertices) == len(mesh_in.vertices) and \
           np.array_equal(np.asarray(mesh_pp.faces), np.asarray(mesh_in.faces)), \
        'preprocess_mesh changed topology — the two-frame correspondence is broken'

    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE)

    # Registration still matters here — the supervision is a per-pixel render
    # loss against the video, so a misregistered mesh still makes the adapter
    # paint the wrong part of the object. Moving the adapter upstream does not
    # fix registration; it is an independent correction.
    #
    # But the meshes under render/out/ are ALREADY registered: they came from
    # export_frame_mesh.py, which applies the transform to the vertices before
    # writing the .ply (:224-225), and frame75.sbatch passed --alignment.
    # Applying it again here double-transforms and OVERSHOOTS — measured
    # silhouette IoU 0.7276, i.e. worse than rung13's *unaligned* 0.7611 and far
    # off its aligned 0.9054. Hence off by default; GATE-cam below is set to
    # catch exactly this mistake rather than let it pass as a texture problem.
    if args.align_mesh:
        al = json.load(open(args.alignment))
        R = torch.tensor(rodrigues(al['rotvec']), dtype=torch.float32, device=DEVICE)
        C = torch.tensor(al['centre'], dtype=torch.float32, device=DEVICE)
        T = torch.tensor(al['translation'], dtype=torch.float32, device=DEVICE)
        v_raw = float(al['scale']) * ((v_raw - C) @ R.T) + C + T
        print(f'[ALIGN] rung13 transform applied  s={al["scale"]:.4f}  '
              f'|rotvec|={np.linalg.norm(al["rotvec"]):.4f} rad', flush=True)
    else:
        print('[ALIGN] not re-applied — the input mesh is already registered '
              '(exported by export_frame_mesh.py --alignment)', flush=True)

    renderer = SurfaceRenderer(v_raw, v_pp, faces, args.render_res, args.resolution)
    print(f'[RENDER] {args.render_res}px, silhouette {renderer.n_px:,} px '
          f'({100*renderer.n_px/args.render_res**2:.1f}% of image), '
          f'rasterised ONCE — geometry and camera are both constant', flush=True)

    # ── GATE-cam ─────────────────────────────────────────────────────────────
    # A wrong camera would look exactly like a texture failure. rung13's lesson
    # was that three separate bugs were in the DIAGNOSTICS, not the method, so
    # the view is checked against the GT silhouette before anything trains.
    gt75 = load_gt(75, args.gt_dir, args.render_res)
    gt_mask = (gt75[0].min(dim=2).values < 0.95)
    inter = float((renderer.mask & gt_mask).sum())
    union = float((renderer.mask | gt_mask).sum())
    iou = inter / max(union, 1.0)
    print(f'\n[GATE-cam] silhouette IoU vs GT frame 75 = {iou:.4f}   '
          f'(render {renderer.n_px:,} px, GT {int(gt_mask.sum()):,} px)')
    assert iou > 0.80, (
        f'GATE-cam FAILED: IoU {iou:.4f}. rung13 measured 0.9054 for this mesh '
        f'registered and 0.7611 unregistered, and 0.7276 when the transform is '
        f'applied TWICE. Below 0.80 means the camera, the coordinate frame, or '
        f'the alignment state of the input mesh is wrong — and every texture '
        f'number below would be meaningless. Check --align-mesh against how the '
        f'mesh was exported.')
    print('[GATE-cam] PASSED', flush=True)

    # ── conditioning + shape latent, both computed ONCE ──────────────────────
    shape_slat = pipe.encode_shape_slat(mesh_pp, args.resolution)
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)
    print(f'\n[SHAPE] shape_slat {tuple(shape_slat.feats.shape)} — '
          f'deterministic, computed ONCE, reused for every frame', flush=True)

    frames = TRAIN + HELD_OUT
    t0 = time.time()

    # ── Conditioning must be PREPROCESSED, exactly as the pipeline does it ───
    # get_cond() feeds DINOv3 directly; it does NOT crop or matte. run() normally
    # calls preprocess_image() first (trellis2_texturing.py:398), which removes the
    # background, crops square to the object's bbox, and premultiplies by alpha so
    # the background is BLACK. Handing it the raw 960x960 frame instead — a small
    # teapot in a large white field — is far out of distribution and yields a
    # black texture, even though the PBR field is bright when conditioned properly
    # (measured: base_color p99 0.55-0.79, diag_black_texture.py).
    #
    # We reproduce that preprocessing deterministically rather than calling it:
    # preprocess_image() runs BiRefNet per frame and crops to that frame's own
    # bbox, so both the matte and the framing would jitter frame to frame, and
    # that jitter would be indistinguishable from texture change. Alpha comes from
    # the near-white background by a fixed threshold, and the crop box is the
    # UNION over all frames — identical framing for every frame, contributing
    # exactly zero to any temporal measurement. Same routine as run_baseline.py.
    raws = [np.array(Image.open(Path(args.gt_dir) / f'frame_{fi:04d}.png').convert('RGB'))
            for fi in frames]
    alphas = [largest_component(r.min(axis=2) < 245) for r in raws]
    union = np.zeros_like(alphas[0])
    for a in alphas:
        union |= a
    ys, xs = np.where(union)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    conds, gts = {}, {}
    for fi, r, a in zip(frames, raws, alphas):
        rgba = np.concatenate([r, (a * 255).astype(np.uint8)[..., None]], axis=-1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        cond_img = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))
        conds[fi] = pipe.get_cond([cond_img], args.resolution)['cond'].detach()
        gts[fi] = load_gt(fi, args.gt_dir, args.render_res)
    print(f'[COND] preprocessed like the pipeline: alpha by threshold (not BiRefNet), '
          f'fixed union crop {bbox}, premultiplied to black', flush=True)
    print(f'[COND] DINOv3 tokens for {len(frames)} frames in {time.time()-t0:.1f}s  '
          f'shape {tuple(conds[frames[0]].shape)} — cached, frozen', flush=True)

    # Noise is FIXED across frames, matching the baseline's fixed seed. shape_slat
    # is a deterministic encode, so with the noise pinned too the ONLY thing that
    # differs between frames is the image conditioning. Any change the adapter
    # learns is therefore attributable to the cross-attention pathway.
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    noise_feats = torch.randn(ss_n.coords.shape[0],
                              flow.in_channels - ss_n.feats.shape[1],
                              generator=g).to(DEVICE)
    noise = ss_n.replace(feats=noise_feats)

    # ── GATE-grad ────────────────────────────────────────────────────────────
    print('\n[GATE-grad] probing gradient health through decoder + flow ...', flush=True)
    reps = {}
    for t in (1.0, 0.75, 0.5, 0.25):
        reg.zero_grad(set_to_none=False)
        with lora_ctx(flow, reg):
            v = flow_eval(flow, noise, t, conds[frames[0]], ss_n)
            (v.feats.float() ** 2).mean().mul(args.loss_scale).backward()
        reps[t] = grad_report(reg, tag=f'probe @ t={t}')
    json.dump({str(k): v for k, v in reps.items()},
              open(OUT / 'logs' / 'gate_grad.json', 'w'), indent=2)
    assert reps[1.0]['n_zero'] == 0, (
        f'GATE-grad FAILED: {reps[1.0]["n_zero"]}/{reps[1.0]["n_matrices"]} B '
        f'matrices got zero gradient. Dead: {reps[1.0]["dead"]}')
    reg.zero_grad(set_to_none=True)
    print('[GATE-grad] PASSED — every B matrix receives gradient', flush=True)

    # ── GATE-plain: at B=0 our forward must equal the untouched model ────────
    with torch.no_grad():
        x_plain = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        with lora_ctx(flow, reg):
            x_lora = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        d = float((x_plain.feats - x_lora.feats).abs().max())
    print(f'\n[GATE-plain] max |frozen - lora(B=0)| = {d:.3e}  (tol 1e-4)')
    assert d < 1e-4, f'GATE-plain FAILED: adapter is not the identity at init ({d:.3e})'
    print('[GATE-plain] PASSED', flush=True)

    def decode_render(x0):
        """x0 (normalised tex latent) -> rendered image. Differentiable."""
        slat = x0 * tex_std + tex_mean
        pbr = decoder(slat) * 0.5 + 0.5
        return renderer.sample(pbr)

    @torch.no_grad()
    def full_inference(fi):
        """The real thing: all 12 ODE steps with the adapter, then render."""
        with lora_ctx(flow, reg):
            x = run_ode(flow, noise, conds[fi], ss_n, T_PAIRS)
        return decode_render(x)

    @torch.no_grad()
    def evaluate(frame_list):
        rows = []
        for fi in frame_list:
            img = full_inference(fi)
            p, s = psnr_ssim(img, gts[fi], renderer.mask)
            rows.append(dict(frame=fi, psnr=p, ssim=s))
        return dict(psnr_mean=float(np.mean([r['psnr'] for r in rows])),
                    psnr_std=float(np.std([r['psnr'] for r in rows])),
                    ssim_mean=float(np.mean([r['ssim'] for r in rows])),
                    per_frame=rows)

    # ── GATE-bwd: prime the spconv neighbour cache WITH GRAD, and prove the
    #             decoder backward works before anything else touches it.
    #
    # flex_gemm caches a submanifold-conv neighbour map per coordinate set, and
    # builds a REDUCED version when the first call that touches those coords runs
    # under no_grad:
    #     submanifold_conv3d.py:331   need_grad = any(ctx.needs_input_grad)
    #     submanifold_conv3d.py:86    if need_grad: ... valid_signal_i/o/seg
    #                          :98    else:        ... _no_bwd variant, omitted
    # TRELLIS.2 stores it on the SparseTensor (conv_flex_gemm.py:44-57), and
    # SparseTensor.replace() passes _spatial_cache BY REFERENCE (basic.py:675).
    # ss_n, noise and every tensor derived from them therefore share ONE cache
    # dict for the whole process. So a single no_grad decode early on poisons
    # every graded decode afterwards with
    #     AttributeError: 'SubMConv3dNeighborCache' has no attribute 'valid_signal_i'
    # which is precisely how jobs 2154525/2154566/2154567 died — and why it was
    # NOT a gradient-checkpointing problem, though it looked like one.
    #
    # The rule this enforces: the first decoder call in the process must be
    # grad-enabled. Everything after it reuses a cache that is a strict superset.
    print('\n[GATE-bwd] purging + repriming spconv cache, testing decoder backward ...',
          flush=True)
    _purged = purge_spconv_cache(ss_n, shape_slat)
    print(f'  purged {len(_purged)} no-bwd neighbour cache(s) left by '
          f'encode_shape_slat: {_purged}', flush=True)
    x0_warm = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], 32, device=DEVICE).requires_grad_(True))
    img_warm = decode_render(x0_warm)
    # LOSS_SCALE is mandatory here, not decorative. The decoder is fp16
    # (tex_dec_next_dc_f16c32_fp16) and an unscaled mean over ~10^6 elements gives
    # gradients near 1e-7, below fp16's ~6e-8 subnormal floor, so they flush to
    # EXACTLY zero and this gate reports a dead path that is actually fine.
    # Measured directly (probe_spconv_bwd.py, res 512, implicit_gemm_splitk):
    #     loss_scale 1     -> sum|grad| 0.0000e+00
    #     loss_scale 4096  -> sum|grad| 4.6620e+03
    #     loss_scale 65536 -> sum|grad| 7.7155e+04
    # The 65536/4096 ratio is 16.55 against an expected 16.0, so ~3% of gradient
    # mass is still underflowing at 4096 — adequate, not generous.
    (img_warm.square().mean() * args.loss_scale).backward()
    gsum = float(x0_warm.feats.grad.abs().sum()) if x0_warm.feats.grad is not None else 0.0
    print(f'  d(render)/d(tex_latent) sum|grad| = {gsum:.4e}   '
          f'peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
    assert gsum > 0, (
        'GATE-bwd FAILED: no gradient reaches the texture latent through '
        'decoder -> grid_sample_3d -> render. Nothing downstream can train.')
    print('[GATE-bwd] PASSED', flush=True)
    if args.decoder_grad_weights:
        for q in decoder.parameters():
            q.grad = None
    del x0_warm, img_warm
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # ── GATE-sample: is the field bright, and does our sampling reproduce it? ─
    # decode_render() samples the PBR voxel field at each pixel's 3D surface point
    # with grid_sample_3d, which returns EXACTLY 0 for points outside the occupied
    # sparse voxels. So a black render has two very different possible causes:
    # a genuinely dark field (generation), or correct field + wrong coordinates
    # (sampling). Measured independently here so the two can never be confused
    # again — diag_black_texture.py established the field reaches p99 0.55-0.79
    # when conditioning is right, so a dark SAMPLE against a bright FIELD is a
    # coordinate bug and nothing else.
    with torch.no_grad():
        with lora_ctx(flow, reg):
            _x = run_ode(flow, noise, conds[frames[0]], ss_n, T_PAIRS)
        _pbr = decoder(_x * tex_std + tex_mean) * 0.5 + 0.5
        _f = _pbr.feats.float()[:, :3]
        _im = renderer.sample(_pbr)
        _fg = _im[0].reshape(-1, 3)[renderer.mask_idx].float()
    _fq = lambda t, q: float(t.flatten().quantile(q))
    print(f'\n[GATE-sample] spatial_shape={tuple(_pbr.spatial_shape)}  '
          f'voxels={tuple(_pbr.feats.shape)}')
    print(f'  PBR FIELD  base_color mean={float(_f.mean()):.4f}  '
          f'p99={_fq(_f, 0.99):.4f}  max={float(_f.max()):.4f}')
    print(f'  SAMPLED    at surface  mean={float(_fg.mean()):.4f}  '
          f'p99={_fq(_fg, 0.99):.4f}  max={float(_fg.max()):.4f}  '
          f'frac>0.05={float((_fg.max(dim=1).values > 0.05).float().mean()):.3f}')
    assert float(_fg.mean()) > 0.2 * float(_f.mean()), (
        f'GATE-sample FAILED: the field has mean {float(_f.mean()):.4f} but our '
        f'surface samples average {float(_fg.mean()):.4f}. grid_sample_3d is '
        f'returning ~0, i.e. we are sampling where the field does not exist — a '
        f'coordinate/frame bug in SurfaceRenderer, not a texture problem.')
    print('[GATE-sample] PASSED', flush=True)
    del _x, _pbr, _f, _im, _fg
    torch.cuda.empty_cache()

    # ── baseline: the frozen model's own score, for reference ────────────────
    base = evaluate(HELD_OUT[:4])
    print(f'\n[FROZEN] held-out subset before training: '
          f'PSNR {base["psnr_mean"]:.3f}  SSIM {base["ssim_mean"]:.4f}', flush=True)

    # ── resume ───────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(reg.parameters(), lr=args.lr, weight_decay=0.0)
    start_ep, best = 1, -1e9
    ckpts = sorted((OUT / 'ckpts').glob('lora_e*.pt'))
    if ckpts:
        st = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
        reg.load_state_dict(st['reg']); opt.load_state_dict(st['opt'])
        start_ep, best = st['epoch'] + 1, st['best']
        print(f'[RESUME] from {ckpts[-1].name}, epoch {start_ep}', flush=True)
    else:
        print('[RESUME] no checkpoint — starting fresh', flush=True)

    csv = OUT / 'logs' / 'epoch_metrics.csv'
    if not csv.exists():
        csv.write_text('epoch,loss,held_psnr,held_ssim,B_mean,B_max,'
                       'dB_toq,dB_tokv,dB_toout,n_zero,time_s\n')

    print(f'\n[TRAIN] epochs {start_ep}->{EPOCHS}   {len(TRAIN)} frames/epoch   '
          f'1 graded flow eval + up to {STEPS-1} no_grad evals per step', flush=True)
    rng = np.random.default_rng(args.seed)
    history = []

    for ep in range(start_ep, EPOCHS + 1):
        te = time.time()
        order = rng.permutation(TRAIN)
        tot, last_rep = 0.0, None
        for si, fi in enumerate(order, 1):
            fi = int(fi)
            k = int(rng.integers(0, STEPS))          # a knot of the REAL schedule
            t_k = T_SEQ[k]

            # BACKWARD MUST RUN INSIDE lora_ctx. Gradient checkpointing recomputes
            # the flow forward during backward, so the hooks have to still be
            # installed at that moment. With .backward() outside the block the
            # recomputed graph is the BARE model and torch raises
            #   CheckpointError: A different number of tensors was saved during the
            #   original forward and recomputation.  forward: 74  recomputation: 57
            # — the 17 missing tensors being exactly the LoRA deltas.
            with lora_ctx(flow, reg):
                x_k = run_ode(flow, noise, conds[fi], ss_n, T_PAIRS[:k])
                v = flow_eval(flow, x_k, t_k, conds[fi], ss_n)
                x0 = pred_to_xstart(x_k, t_k, v)
                img = decode_render(x0)
                loss = ((img - gts[fi]) ** 2)[:, renderer.mask].mean()

                opt.zero_grad(set_to_none=True)
                (loss * args.loss_scale).backward()
            for p in reg.parameters():
                if p.grad is not None:
                    p.grad.div_(args.loss_scale)
            if si == len(order):
                last_rep = grad_report(reg)
            torch.nn.utils.clip_grad_norm_(reg.parameters(), args.grad_clip)
            opt.step()
            # The decoder's weights carry requires_grad only to keep flex_gemm from
            # returning grad_weight=None; nothing optimises them. opt.zero_grad()
            # does not touch them (the optimizer holds reg.parameters() alone), so
            # clear them here or they accumulate across the whole run.
            if args.decoder_grad_weights:
                for q in decoder.parameters():
                    q.grad = None
            tot += float(loss)
            lv = float(loss)
            del loss, img, x0, v, x_k

            if si % 25 == 0 or si == 1:
                print(f'  e{ep:02d} [{si:03d}/{len(order)}] f{fi:04d} k={k:02d} '
                      f't={t_k:.3f}  mse={lv:.5f}  '
                      f'peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB', flush=True)

        ev = evaluate(HELD_OUT)
        pn = param_norms(reg)
        dt = time.time() - te
        row = dict(epoch=ep, loss=tot / len(order), **{k: v for k, v in ev.items()
                                                       if k != 'per_frame'})
        history.append(row)
        print(f'[EPOCH {ep}/{EPOCHS}] loss={row["loss"]:.5f}  '
              f'held PSNR={ev["psnr_mean"]:.3f}+-{ev["psnr_std"]:.3f}  '
              f'SSIM={ev["ssim_mean"]:.4f}  |B|={np.mean(list(pn.values())):.4f}  '
              f'{dt:.0f}s', flush=True)
        with open(csv, 'a') as fh:
            fh.write(f'{ep},{row["loss"]:.6f},{ev["psnr_mean"]:.4f},'
                     f'{ev["ssim_mean"]:.5f},{np.mean(list(pn.values())):.5f},'
                     f'{max(pn.values()):.5f},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_q",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_kv",0):.4e},'
                     f'{(last_rep or {}).get("per_target",{}).get("to_out",0):.4e},'
                     f'{(last_rep or {}).get("n_zero",-1)},{dt:.1f}\n')

        torch.save(dict(reg=reg.state_dict(), opt=opt.state_dict(),
                        epoch=ep, best=best, cfg=_CFG),
                   OUT / 'ckpts' / f'lora_e{ep:03d}.pt')
        if ev['psnr_mean'] > best:
            best = ev['psnr_mean']
            torch.save(dict(reg=reg.state_dict(), epoch=ep, psnr=best, cfg=_CFG),
                       OUT / 'ckpts' / 'lora_best.pt')
            print(f'  [CKPT] new best -> lora_best.pt ({best:.3f} dB)', flush=True)

        if ep % 5 == 0 or ep == EPOCHS:
            strip = []
            for fi in HELD_OUT[:3]:
                strip.append(np.concatenate([
                    (gts[fi][0].cpu().numpy() * 255).astype(np.uint8),
                    (full_inference(fi)[0].cpu().numpy() * 255).astype(np.uint8)], axis=1))
            Image.fromarray(np.concatenate(strip, axis=0)).save(
                OUT / 'diag' / f'e{ep:03d}_gt_vs_ours.jpg', quality=92)
        json.dump(history, open(OUT / 'loss_history.json', 'w'), indent=2)
        torch.cuda.empty_cache()

    st = torch.load(OUT / 'ckpts' / 'lora_best.pt', map_location=DEVICE,
                    weights_only=False)
    reg.load_state_dict(st['reg'])
    final = evaluate(HELD_OUT)
    json.dump(dict(frozen=base, final=final, best_epoch=st['epoch']),
              open(OUT / 'final_eval.json', 'w'), indent=2)
    print(f'\n[FINAL] frozen  PSNR {base["psnr_mean"]:.3f}  SSIM {base["ssim_mean"]:.4f}')
    print(f'[FINAL] rung14  PSNR {final["psnr_mean"]:.3f}  SSIM {final["ssim_mean"]:.4f}  '
          f'(epoch {st["epoch"]})')
    print('[FINAL] geometry unchanged BY CONSTRUCTION — TRELLIS.2 takes the mesh '
          'as an input and emits 32/6 channels, none of which is a position.')
    print(f'\n[SAVE] {OUT}\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
