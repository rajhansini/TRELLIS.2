# Ground truth for the method section — rung27 + MCFM (temporal_only)

Pulled from the code, not from memory. Every number here is either read off a
checkpoint or quoted from the source line that produces it.

## 1. What the adapter touches

`--targets qkvo+sa --blocks all --rank 4`, LoRA on **all 30 blocks** of
`tex_slat_flow_model_512`. Per block, five adapters (shapes read from
`runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94/ckpts/lora_best.pt`):

| adapter | on | A | B | note |
|---|---|---|---|---|
| `to_q`    | cross_attn | (4, 1536) | (1536, 4) | |
| `to_kv`   | cross_attn | (4, 1024) | (3072, 4) | **fused** K and V — one rank-4 A shared by both |
| `to_out`  | cross_attn | (4, 1536) | (1536, 4) | |
| `sa_qkv`  | self_attn  | (4, 1536) | (4608, 4) | **fused** Q, K, V — `Linear(ch, 3*ch)` |
| `sa_out`  | self_attn  | (4, 1536) | (1536, 4) | |

So `model_channels = 1536` and `cond_channels = 1024`. **Total 2,334,720 trainable
parameters.** The MLP is untouched.

Two things a method section commonly gets wrong here:
- `to_kv` is **fused**. A rank-4 A is shared by the K and V rows, so their updates
  are forced through one 4-dimensional subspace and cannot be separated post hoc.
  Do not describe them as independent K and V adapters.
- `sa_qkv` is likewise one fused `Linear(ch, 3*ch)`, not three adapters.

## 2. The LoRA form

`delta = (alpha / rank) * B @ (A @ x)`, A Kaiming-init, **B zero-init** so step 0 is
exactly the identity. With `rank = 4, alpha = 4` the scaling factor is **exactly
1.0**. If the paper reports alpha/rank as a tuned hyperparameter, it is not — it is
set so that a rank sweep does not confound rank with effective step size
(`dW = B @ A` sums `rank` outer products, so without the factor a larger rank means
a larger update at the same learning rate).

## 3. MCFM, temporal_only

`temporal_only_w3` is an alias for **`v2_D`** — same string reaches the config and
the run-directory hash, so the two names are the same run.

- **Parameter-free.** `blend_conds` is decorated `@torch.no_grad()`. Nothing in
  MCFM is trained. If the method section calls it a learned temporal-attention
  module, that is wrong.
- **It runs before the flow model.** MCFM rewrites the cached DINOv3 image-
  conditioning tokens; it is not a layer inserted into the network.
- **v2 = per-position.** Token *i* of frame *t* attends over **token *i* only** in
  the window — 1029 independent softmaxes over W items. There is no spatial term;
  spatial mixing is left to the model's own cross-attention downstream. (v3, the
  joint variant, pools all W*N tokens into one softmax — that is a *different* arm.)
- **Window.** `_D` = offsets `(-1, 0, +1)`, so **W = 3**. (`_C` = `(0, +1)`, W = 2.)
- **Edges clamp**, they do not wrap: frame 1's `t-1` is frame 1, so the first and
  last frames blend over a truncated window.
- **Scale** is `D**-0.5` with D = 1024 (token dim), standard scaled dot-product.

Exact operation, from `blend_window`:

    stack = torch.stack(window_toks, dim=1)          # (N, W, D)
    q     = cur.unsqueeze(1)                         # (N, 1, D)
    attn  = softmax(q @ stack.transpose(1,2) * D**-0.5, dim=-1)
    out   = (attn @ stack).squeeze(1)                # (N, D)

## 4. Token counts — the easiest number to get wrong

| pipeline | encoder | tokens @ res 512 |
|---|---|---|
| TRELLIS **1** | DINOv2 | 1374 |
| TRELLIS **.2** | DINOv3 | **1029** |

This work is TRELLIS.2, so **1029**. 1374 is a DINOv2-at-518 number and belongs to
the TRELLIS 1 validation runs only. At resolution 1024 it is 4101.

MCFM has no learned parameters, so the *operation* transfers between the two
pipelines but the *tokens* do not.

## 5. Sampling / training timesteps

From `flow_euler.py:114-118`: `STEPS = 12`, `RESCALE_T = 3.0`,
`t = 3t' / (1 + 2t')` over `linspace(1, 0, 13)`. The knots are **front-loaded** —
seven of the twelve sit above t = 0.75. Training timesteps are drawn from **this
discrete list**, not from a uniform [0, 1], so the adapter is never trained at
timesteps inference never visits. If the paper says "timesteps sampled uniformly",
that is wrong.

## 6. Training configuration

`--recon l1 --lpips --w-lpips 0.1 --epochs 30 --seed 42 --lr 1e-4`,
`--render-res 960 --resolution 512`. 150 frames (121 for pumpkin_rot).
Loss = L1 + 0.1 * LPIPS, and in the logs the LPIPS term runs 9–15% of total loss.

## 7. Mutual exclusions

`--mcfm` and `--context-window` are asserted mutually exclusive: MCFM pre-collapses
the window that `--context-window` exists to hand over intact. A described method
that both blends *and* concatenates is not a configuration the trainer can produce.
