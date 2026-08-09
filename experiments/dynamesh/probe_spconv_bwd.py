"""
probe_spconv_bwd.py — which flex_gemm sparse-conv config can backprop through
TRELLIS.2's texture decoder with FROZEN weights?

WHY THIS EXISTS
  TRELLIS.2's texturing pipeline is inference-only (`run()` is @torch.no_grad()),
  so no shipped code path ever backprops through tex_slat_decoder. rung14 must.
  Two independent defects show up when you try, and they interact:

  1. Cache variant. The neighbour cache is built WITHOUT valid_signal_i/o/seg
     when the first call on a coordinate set does not require grad
     (submanifold_conv3d.py:331, :86-101). encode_shape_slat always runs first
     and always frozen, and SparseTensor.replace() shares _spatial_cache by
     reference (basic.py:675), so the reduced cache reaches every later decode.
     The masked_* backward reads those fields unconditionally -> AttributeError.

  2. Frozen weights. kernels/triton/.../bwd_implicit_gemm.py:166-169 returns
     grad_weight=None when weight.requires_grad is False, but the caller does
     `grad_weight = grad_weight.reshape(...)` with no None check
     (submanifold_conv3d.py:270) -> AttributeError: 'NoneType' has no 'reshape'.
     Only the EXPLICIT_GEMM branch (:229-260) guards with `if weight.requires_grad`.

  So the candidate fixes are: use explicit_gemm, or leave the fast algorithm and
  make the decoder weights require grad so grad_weight is never None (paying for
  a gradient we discard). This script measures which combinations actually run
  and what they cost, instead of reasoning about it a sixth time.

Usage (compute node only — GLIBC >= 2.32):
  python probe_spconv_bwd.py --algo explicit_gemm
  python probe_spconv_bwd.py --algo implicit_gemm --grad-weights
"""

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')

ap = argparse.ArgumentParser()
ap.add_argument('--algo', required=True)
ap.add_argument('--grad-weights', action='store_true',
                help='let decoder weights require grad so grad_weight is never None')
ap.add_argument('--mesh', default='/net/projects/ranalab/rajhansini/TRELLIS/'
                                  'render/out/f0075/frozen_f0075.ply')
ap.add_argument('--resolution', type=int, default=1024)
ap.add_argument('--loss-scale', type=float, default=4096.0,
                help='must match training: the fp16 decoder underflows without it')
ap.add_argument('--out', default=None)
args = ap.parse_args()

import numpy as np
import torch
import torch.nn as nn
import trimesh

DEV = torch.device('cuda')
res = dict(algo=args.algo, grad_weights=args.grad_weights,
           resolution=args.resolution, ok=False, error=None,
           peak_gib=None, grad_sum=None, seconds=None)

try:
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config

    conv_config.FLEX_GEMM_ALGO = args.algo
    print(f'[PROBE] algo={args.algo}  grad_weights={args.grad_weights}  '
          f'resolution={args.resolution}', flush=True)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    dec = pipe.models['tex_slat_decoder'].to(DEV)

    # The realistic setting: everything frozen except what --grad-weights changes.
    for m in pipe.models.values():
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)
    if args.grad_weights:
        for p in dec.parameters():
            p.requires_grad_(True)

    mesh = trimesh.load(args.mesh, process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh)
    # This runs frozen and non-grad, which is exactly what poisons the cache.
    shape_slat = pipe.encode_shape_slat(mesh_pp, args.resolution)
    print(f'[PROBE] shape_slat {tuple(shape_slat.feats.shape)}', flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    x0 = shape_slat.replace(feats=torch.randn(
        shape_slat.coords.shape[0], 32, device=DEV).requires_grad_(True))
    out = dec(x0) * 0.5 + 0.5
    print(f'[PROBE] decoded -> {tuple(out.feats.shape)}  forward ok', flush=True)
    # The decoder is fp16 (tex_dec_next_dc_f16c32_fp16). An unscaled mean over
    # millions of elements gives gradients ~1e-7, under fp16's ~6e-8 subnormal
    # floor, so they flush to EXACTLY zero — which reads as a dead kernel but is
    # just underflow. This is the same failure that cost TRELLIS v1 weeks in
    # dec_mesh. Training applies LOSS_SCALE=4096, so the probe must too or it
    # measures the wrong thing.
    (out.feats.square().mean() * args.loss_scale).backward()

    gs = float(x0.feats.grad.abs().sum()) if x0.feats.grad is not None else 0.0
    res.update(ok=gs > 0, grad_sum=gs, seconds=round(time.time() - t0, 2),
               peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(f'[PROBE] {"OK" if gs > 0 else "ZERO-GRAD"}  sum|grad|={gs:.4e}  peak={res["peak_gib"]} GiB  '
          f'{res["seconds"]}s', flush=True)

except Exception as e:
    res['error'] = f'{type(e).__name__}: {e}'
    res['peak_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f'[PROBE] FAILED  {res["error"]}', flush=True)

print('[RESULT] ' + json.dumps(res), flush=True)
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=2))
