# TRELLIS.2 texturing — the complete workflow, and where rung14's LoRA goes

Everything here is read out of `/net/projects/ranalab/rajhansini/TRELLIS.2`, not
recalled. File and line references are to that tree. Checkpoint configs are read from
`/net/scratch/rajhansini/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B`.

---

## 0. The device model — read this first, it has already cost two jobs

`texturing_pipeline.json` does not set `low_vram`, and
`Trellis2TexturingPipeline.__init__` defaults it to `True`. That changes what `.cuda()`
means:

```python
# trellis2_texturing.py:97-103   — OVERRIDES pipelines/base.py:64-69
def to(self, device):
    self._device = device
    if not self.low_vram:          # <- False, so the body never runs
        super().to(device)
        self.image_cond_model.to(device)
        ...
```

So `pipe.cuda()` **sets a string and moves nothing**. Each model is walked to the GPU
on demand and walked back to CPU immediately after:

| method | moves | then |
|---|---|---|
| `get_cond` | `image_cond_model` | `.cpu()` |
| `encode_shape_slat` | `shape_slat_encoder` | `.cpu()` |
| `sample_tex_slat` | `flow_model` | `.cpu()` |
| `decode_tex_slat` | `tex_slat_decoder` | `.cpu()` |

`pipe.run()` therefore works. **Anything that calls a model directly does not** — it
sees CUDA activations against CPU weights. That is exactly how job 2153281 died at
`input_layer`. rung14 calls the flow directly, so it must pin the models itself:

```python
flow = pipe.models[f'tex_slat_flow_model_{RES}']
flow.to(DEVICE)                                  # required
pipe.models['tex_slat_decoder'].to(DEVICE)       # required once training renders
```

And it must keep them pinned — never call `pipe.run()` in the same process afterwards,
because that would move them back to CPU underneath you.

---

## 1. The inference pipeline, stage by stage

`Trellis2TexturingPipeline.run()` — `trellis2_texturing.py:377-411`. Six stages.

```
INPUT: one trimesh + one PIL image
   │
   ▼
[A] preprocess_image                                     L124-159
    RGBA with real alpha?  -> used directly
    else                   -> BiRefNet (briaai/RMBG-2.0) removes the background
    crop to the alpha bbox, squared and centred
    premultiply RGB by alpha  -> black background
    NOTE: this is a per-frame neural step. Its mask jitters frame to frame.
          For a temporal experiment, derive alpha yourself and pass
          preprocess_image=False, or that jitter is counted as texture change.
   │
   ▼
[B] preprocess_mesh                                      L105-122
    normalise into [-0.5, 0.5]:  v = (v - centre) * 0.99999 / max_extent
    axis permute: y <- -z, z <- y
    faces UNCHANGED.  UVs preserved if mesh.visual.uv exists.
    This is a similarity transform. The shape is not altered.
   │
   ▼
[C] get_cond                                             L161-183
    DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m)
    image resized to `resolution` (512 or 1024)
    -> patch tokens, layer-normed          [B, N, 1024]
    neg_cond = zeros_like(cond)
    cond is the ONLY path by which the image reaches the 3D model.
    Accepts a LIST of images -> VarLenTensor. Multi-view is plumbed through.
   │
   ▼
[D] encode_shape_slat                                    L185-224
    o_voxel.convert.mesh_to_flexible_dual_grid(v, f, grid_size=resolution,
        aabb=[-0.5,-0.5,-0.5]..[0.5,0.5,0.5],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
      -> voxel_indices, dual_vertices, intersected
    feats = dual_vertices * resolution - voxel_indices   (sub-voxel offsets)
    -> shape_slat_encoder -> SparseTensor [N_vox, 32]
    DETERMINISTIC. No noise anywhere in this stage. Same mesh -> same latent,
    bit for bit. Cache it once and reuse for every frame.
    (measured on frozen_f0075.ply at resolution 1024:  N_vox = 9,204)
   │
   ▼
[E] sample_tex_slat                                      L226-267
    shape_slat normalised by shape_slat_normalization (mean/std, 32 numbers each)
    noise ~ N(0,1)  [N_vox, 64-32 = 32]
    FlowEulerGuidanceIntervalSampler.sample(
        flow, noise, concat_cond=shape_slat_norm, cond=..., neg_cond=...,
        steps=12, guidance_strength=1.0, guidance_interval=[0.6,0.9],
        rescale_t=3.0)
    output denormalised by tex_slat_normalization
    -> tex_slat [N_vox, 32]
   │
   ▼
[F] decode_tex_slat                                      L269-287
    tex_slat_decoder(slat) * 0.5 + 0.5   -> PBR voxels [N_fine, 6]
      [0:3] base_color   [3:4] metallic   [4:5] roughness   [5:6] alpha
   │
   ▼
[G] postprocess_mesh                                     L289-374
    UVs: reuse mesh.visual.uv if present, ELSE cumesh.uv_unwrap (splits seams,
         changes the vertex/face arrays, and on this cluster raises CUDA 209 —
         supply UVs and it is never called)
    rasterize UV space at texture_size (nvdiffrast, GL context)
    interpolate 3D position per texel
    flex_gemm.ops.grid_sample.grid_sample_3d(pbr_voxel, grid=pos, trilinear)
    cv2.inpaint across seams
    -> trimesh with PBRMaterial(baseColorTexture, metallicRoughnessTexture)
```

**The contract:** geometry in, geometry out. `out_channels` is 32 for the flow and 6 for
the decoder — neither can emit a vertex position. The mesh is a condition, not a
prediction.

---

## 2. The flow model, and the t schedule

`slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json`:

```
resolution       64        in_channels   64     = 32 noisy tex latent + 32 shape latent
model_channels   1536      out_channels  32       texture only
cond_channels    1024      num_blocks    30
num_heads        12        mlp_ratio     5.3334
pe_mode          rope      share_mod     true
qk_rms_norm      true      qk_rms_norm_cross  true
dtype            bfloat16
```

`SLatFlowModel.forward` — `models/structured_latent_flow.py:169-200`:

```python
x = sp.sparse_cat([x, concat_cond], dim=-1)      # 32 + 32 -> 64   shape enters HERE
h = self.input_layer(x)                          # 64 -> 1536
h = manual_cast(h, self.dtype)                   # -> bf16
t_emb = self.adaLN_modulation(self.t_embedder(t)) # share_mod=True
cond = manual_cast(cond, self.dtype)
for block in self.blocks:                        # 30 x ModulatedSparseTransformerCrossBlock
    h = block(h, t_emb, cond)
h = manual_cast(h, x.dtype)                      # back to fp32
h = layer_norm(h); h = self.out_layer(h)         # 1536 -> 32
```

Shape conditions by **concatenation**. Image conditions by **cross-attention**. Those
are two structurally different channels and only the second is worth adapting.

### The timestep schedule

`FlowEulerSampler.sample` — `flow_euler.py:114-118`:

```python
t_seq = np.linspace(1, 0, steps + 1)
t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)     # rescale_t = 3.0
```

With `steps=12, rescale_t=3.0` the 13 knots are:

```
1.0000  0.9706  0.9375  0.9000  0.8571  0.8077  0.7500
0.6818  0.6000  0.5000  0.3750  0.2143  0.0000
```

Front-loaded: seven of twelve steps live above t=0.75. **Sample training timesteps from
this schedule, not from a uniform [0,1]** — otherwise the LoRA is trained at timesteps
inference never visits.

The model is called with `t -> 1000*t` — `flow_euler.py:44-46`:

```python
def _inference_model(self, model, x_t, t, cond=None, **kwargs):
    t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
    return model(x_t, t, cond, **kwargs)
```

### Why you call the flow directly, not `_inference_model`

`FlowEulerGuidanceIntervalSampler` is `GuidanceIntervalSamplerMixin +
ClassifierFreeGuidanceSamplerMixin + FlowEulerSampler`, and each mixin overrides
`_inference_model` with extra REQUIRED positional args
(`guidance_strength`, `guidance_interval`). Calling it with the base signature raises
`TypeError`.

Worth knowing: with `guidance_strength=1.0`, the CFG mixin short-circuits
(`classifier_free_guidance_mixin.py:10-11`) to a **single cond-only forward pass**. CFG
is a numerical no-op in the shipped texturing config. So calling the flow directly with
`cond` only is not an approximation — it is bit-identical to what inference does, at
half the cost of a naive CFG implementation.

---

## 3. Where the LoRA goes

`ModulatedSparseTransformerCrossBlock` — `modules/sparse/transformer/modulated.py:81`.
Per block, from `modules/sparse/attention/modules.py:62-72`:

```
self_attn  (type="self")    to_qkv  Linear(1536, 4608)     FROZEN   3D propagation
                            to_out  Linear(1536, 1536)     FROZEN
cross_attn (type="cross")   to_q    Linear(1536, 1536)  <- LoRA     what a voxel asks
                            to_kv   Linear(1024, 3072)  <- LoRA     how tokens present
                            to_out  Linear(1536, 1536)  <- LoRA     how it's written back
mlp                         mlp[0]  Linear(1536, 8192)     FROZEN
                            mlp[2]  Linear(8192, 1536)     FROZEN
adaLN_modulation                                           FROZEN
```

Rank 4, all 30 blocks:

```
to_q     4 * (1536 + 1536) = 12,288
to_kv    4 * (1024 + 3072) = 16,384
to_out   4 * (1536 + 1536) = 12,288
                             ------
per block                    40,960     x 30 blocks = 1,228,800 params
                                                      90 B matrices
```

`B` is zero-init, so step 0 is the unmodified model — the same identity guarantee
rung5/rung13 relied on.

**Why cross-attention and not self-attention.** Self-attention is what makes the output
3D-coherent; it is the propagation mechanism the whole bet depends on. The claim is that
steering *what enters* lets TRELLIS.2's own propagation distribute the edit to surface
the camera never saw. Disturbing the propagation itself would defeat the purpose.

**Why not the decoder, where rung13 put it.** The mesh decoder has no cross-attention at
all — the image never reaches it, and all 3D reasoning has finished upstream. Measured on
rung13: 16.8% of vertices supervised, edit strength on unseen vs seen vertices 1.02
(identical), and R² 0.16–0.23 against the training camera's image axis. It learned a
projection, because a projection was the only thing available to it.

Injection is by **forward hook**, so TRELLIS.2's source is never modified:

```python
h = blk.cross_attn.to_q.register_forward_hook(
        lambda m, inp, out: out + lora.to_q(inp[0]).to(out.dtype))
```

---

## 4. The training step

```
ONCE, cached
  mesh ──► preprocess_mesh ──► encode_shape_slat ──► shape_slat [9204, 32]
  frames 1..150 ──► DINOv3 ──► cond tokens [1, N, 1024]  (one per frame)

PER STEP, frame t
  ┌──────────────────────────────────────────────────────────────────────┐
  │ k ~ the 12-knot rescaled schedule (NOT uniform)                       │
  │                                                                       │
  │ x_k ← integrate the ODE from t=1 down to k        no_grad, ~6 evals   │
  │       (Euler, x ← x − (t − t_prev) · v, flow called directly)          │
  │                                                                       │
  │ v   ← flow(x_k, 1000·k, cond, concat_cond=shape_slat_norm)            │
  │                                                    WITH grad, 1 eval  │
  │       30 blocks: self_attn frozen, cross_attn + LoRA δ                │
  │                                                                       │
  │ x̂₀  ← (1−σ)·x_k − (σ + (1−σ)·k)·v      = sampler._pred_to_xstart      │
  └──────────────────────────────────────────────────────────────────────┘
        │  denormalise by tex_slat_normalization
        ▼
   tex_slat_decoder (frozen)  ──►  PBR voxels [N_fine, 6]
        │
        ▼
   grid_sample_3d at mesh surface points        differentiable
        │
        ▼
   nvdiffrast render, fixed camera  ──►  loss vs video frame t
        │
        ▼
   gradient reaches ONLY the 90 LoRA B matrices
```

**Why one graded flow evaluation, not twelve.** Backprop through the full ODE means 12
sequential 1.3B forward passes with all activations retained. Drawing `k` fresh every
step means the adapter still sees every timestep across training, at roughly rung13's
cost per step. Using the sampler's own `_pred_to_xstart` guarantees the rectified-flow
parameterisation cannot drift from TRELLIS.2's.

**Supervision is unchanged from rung13** — the video frame, one fixed camera. Only the
adapter's position moved. Note this means **the rung13 registration lesson still
applies**: if the mesh is not aligned to the video object in screen space, a per-pixel
loss still compares two misregistered objects. The alignment transform is still needed;
it is simply no longer entangled with the geometry-freezing question.

---

## 5. Frozen vs trainable

| component | params | state |
|---|---|---|
| `shape_slat_encoder` | — | frozen, run once, cached |
| DINOv3 image encoder | 300M | frozen, run once per frame, cached |
| `tex_slat_flow_model_1024` | 1.3B | frozen |
| `tex_slat_decoder` | — | frozen |
| BiRefNet rembg | — | not used at train time |
| **LoRA on cross_attn** | **1,228,800** | **trainable** |

Geometry cannot change. Not by assertion, by architecture — the mesh is an input and
nothing downstream emits vertex positions. No two-pass splice, no per-frame vertex-count
check needed.

---

## 6. Gates — fail in 60 s, not in 4 h

| gate | asserts | catches |
|---|---|---|
| `GATE 0` | param count == rank × (12288+16384+12288)/4 × 30, and every B is exactly 0 | mis-built registry; non-identity init |
| `GATE-grad` | none of the 90 B matrices gets zero gradient, probed at t = 1.0/0.75/0.5/0.25 | bf16 underflow behind the softmax — the TRELLIS v1 fp16 FTZ failure, which cost weeks |
| `GATE-device` | every model used directly reports a CUDA device | the `low_vram` no-op `.to()` |

`GATE-grad` is reported on **B, not A**: B is zero-init, so if B's gradient is zero then
A's is zero by the chain rule and nothing can ever move. Probing multiple timesteps
matters because starvation is often timestep-dependent.

**Status: GATE-grad PASSED** (job 2153811, 19:24, 1m17s):

```
0/90 at exactly zero, all four timesteps
t=1.00   to_q 2.05e+01   to_kv 1.33e+02   to_out 2.66e+01   spread 1.4e+02x
t=0.75   to_q 3.02e+00   to_kv 2.16e+01   to_out 4.15e+00   spread 2.2e+02x
t=0.50   to_q 2.11e+00   to_kv 1.97e+01   to_out 2.85e+00   spread 1.5e+03x
t=0.25   to_q 2.09e+00   to_kv 2.09e+01   to_out 2.89e+00   spread 2.9e+03x
```

`to_kv` carries 5–6× the others throughout — that is the image→3D projection, the one
that most needs to be alive. Watch the spread: it widens to 2.9e+03× at low t, driven by
block 0 (2.9e+02 against a ~1e+01 typical). Nothing is dead, but per-block gradient
scaling is worth revisiting once training runs.

`LOSS_SCALE = 4096` is carried over from the v1 fix and is still applied.

---

## 7. What decides success

Not training-view PSNR — rung13 already scores 22.6 dB there and is still wrong off-axis.

1. **Turntable.** Does the texture survive as the camera rotates away from the training
   view? rung13's degrades badly by 180°. This is the whole point of moving the adapter
   upstream of the propagation.
2. **Temporal flicker** frame to frame, against the frozen per-frame TRELLIS.2 baseline
   (`run_baseline.py`) — which is the number that says the problem is open at all.
3. **Geometry unchanged** — free here.

---

## 8. Submitting

```bash
cd /net/projects/ranalab/rajhansini/TRELLIS
export SPCONV_ALGO=native

# gradient probe — passed, rerun only if the model/rank/targets change
sbatch experiments/trellis2_baseline/jobs/rung14_gradprobe.sbatch

# the frozen per-frame comparison this is measured against
sbatch --export=ALL,NFRAMES=150,TAG=frozen_t2 \
       experiments/trellis2_baseline/jobs/baseline.sbatch

# training (once the loop is written)
sbatch experiments/trellis2_baseline/jobs/rung14_train.sbatch
```

Practical notes, all learned the hard way on this cluster:

- `o_voxel`, `cumesh`, `nvdiffrast` and `flex_gemm` need **GLIBC ≥ 2.32**. Login node is
  2.31, compute nodes are 2.35. Everything runs through `sbatch`, no exceptions.
- **`cumesh` raises CUDA 209** ("no kernel image available") on the A40s — built for a
  different arch. Supply UVs (xatlas, CPU) and the code path is never entered.
- Never `sbatch --wrap`: it runs under `/bin/sh` = `dash`, which has no `set -o
  pipefail`. Two jobs died in 0 s before that was worked out.
- `HF_HUB_OFFLINE=1` on compute nodes. Anything not pre-cached fails at load. Currently
  cached and verified: `microsoft/TRELLIS.2-4B` (6.4 GB),
  `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0`.
- Walltime is capped at **4 h** on every partition. Use `--requeue` plus per-epoch
  checkpoints.
