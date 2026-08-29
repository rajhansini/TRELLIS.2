# Paper audit — 3DV '27 submission #92, "DynaMesh"

Audited 2026-08-25 by Raj against the code and run outputs on disk, not against
status docs. Main method throughout is **rung27 + MCFM `v2_D`** (temporal-only,
window `[t-1, t, t+1]`).

Every claim below was checked against a named file and line. Paths are relative to
`/net/projects/ranalab/rajhansini/TRELLIS.2/`, run dirs to `experiments/dynamesh/`.

---

## A. Verified correct — do not touch

| Paper | Source | |
|---|---|---|
| Eq. (1) `α^i = softmax_δ(⟨z_t^i, z_{t+δ}^i⟩ / √D)` | `experiments/dynamesh/mcfm_blend.py`, `blend_window()` variant `v2`: `q = cur.unsqueeze(1)`, `softmax(bmm(q, stack.transpose(1,2)) * scale)`, `scale = D ** -0.5` | exact match |
| L330–332 "each position attends over the same position in the neighboring frames, with the current frame as the query" | `mcfm_blend.py` docstring: "token i of frame t attends over token i of the window only" | ✓ |
| L333 "parameter-free" | `@torch.no_grad()` on `blend_conds()`; no trainable state in the module | ✓ |
| L334–335 default three-frame window `[t−1,t,t+1]` | `_OFFSETS['D'] = (-1, 0, 1)` | ✓ |
| L335 two-frame window `[t,t+1]` | `_OFFSETS['C'] = (0, 1)` | ✓ |
| L336–338 joint variant, "a token may attend to any position in any frame" | `v3`: "all W*N tokens pooled into one softmax" | ✓ |
| L351 `W'x = Wx + s·BAx` | `rung27_selfattn_lora.py:592` — `delta = (alpha/rank) * B @ (A @ x)`, A kaiming-init, B zero-init. s = 4/4 = 1 | ✓ |
| L366–368 rank-4 residuals on query, key–value and output of cross-attention and on the self-attention projections of every generator block | `target_set = ('to_q','to_kv','to_out','sa_qkv','sa_out')`, `active = 0..29`. Train log: `[LORA] 2,334,720 params  150 B matrices (rank 4 x 30 blocks)` = 5 adapters × 30 blocks | ✓ |
| L394–396 L1 plus a small perceptual term, computed inside the silhouette | config `recon='l1'`, `w_lpips=0.1` | ✓ |
| L396–400 "runs the generator to a random point along its denoising trajectory with gradients disabled, takes a single step with gradients enabled" | `rung27_selfattn_lora.py:118` — "run the ODE to a RANDOM timestep k under no_grad (~6 evals avg) / ONE model call at k WITH grad" | exact |
| L416–419 shared sparse structure S and shared denoising noise | `noise` built once from a seeded generator (`rung27:1511-1514`), `ss_n` encoded once; both passed unchanged to every `run_ode(flow, noise, conds[fi], ss_n, T_PAIRS)` | ✓ |
| L454–456 30 epochs, 960² resolution, L1 + 0.1 perceptual, single GPU | config `epochs=30`, `render_res=960`; train log `[RENDER] 960px` | ✓ |

Note for the supplementary: conditioning images are 512 px → **1,029 DINOv3 tokens**
(`mcfm_blend.py` docstring). The paper never states the token count; it is worth one line.

---

## B. Factual errors

### B1 — L353–354. Parameter fraction is wrong under every denominator

Paper: *"the residuals amount to less than a tenth of a percent of the generator's parameters."*

Counted from the safetensors headers in
`models--microsoft--TRELLIS.2-4B/snapshots/af44b45f.../ckpts/`:

| component | params |
|---|---|
| LoRA (ours) | **2,334,720** |
| `slat_flow_imgshape2tex_dit_1_3B_512_bf16` — what we adapt | **1,292,302,880** |
| `tex_dec_next_dc_f16c32_fp16` | 474,216,006 |
| `shape_enc_next_dc_f16c32_fp16` | 354,385,392 |
| full pipeline | 2,120,904,278 |

- against the flow: **0.181 %**
- against the whole pipeline: **0.110 %**

Both are above a tenth of a percent. **Fix: write 0.18 %.**

### B2 — L457. "Sequences are 150 frames" is false for ~18 % of runs

Across all 30-epoch rung27–34 configs: **150 frames → 538 runs, 121 → 114 runs, 135 → 2 runs.**
`penguin_circuits`, `teapot_porcelain` and `teapot_ceramic_crack` are 121 frames;
`ancient_lady` is 135. **Fix: "up to 150 frames".**

### B3 — L489–490. The tLP claim is backwards

Paper: *"The results show that our method has a much lower tLP compared to the baseline
and other ablations."*

`experiments/dynamesh/out/tlp_metric.json` — MCFM's tLP is **higher (worse) on 6 of 8**,
mean **+18.4 %**:

| object | rung27 | +MCFM | Δ |
|---|---|---|---|
| pumpkin_rot | 0.000227 | 0.000413 | +81.9 % |
| penguin_circuits | 0.000960 | 0.001247 | +30.0 % |
| spot_lava | 0.000334 | 0.000431 | +29.0 % |
| horse_metal | 0.000505 | 0.000577 | +14.1 % |
| teapot_ceramic_crack | 0.000369 | 0.000420 | +13.7 % |
| teapot_porcelain | 0.002327 | 0.002513 | +8.0 % |
| teapot_lava2 | 0.000287 | 0.000281 | −1.9 % |
| whale_spots | 0.000616 | 0.000444 | −27.8 % |

The paper's own justification for tLP also defeats itself. L486–488 rejects raw
frame-to-frame difference "because the latter is minimized by an output that does not
move at all" — **tLP is also a first-difference measure and has exactly that failure
mode.** It penalises us for being *slower*, not for being worse.

**Fix — change the metric, do not soften the sentence.** Report the texel triple, which
is already measured on 28 matched object pairs:

- **Temporal Flickering (VBench): −20.0 % mean, lower on 28/28**
- **Jerk (2nd order): −44.4 % mean, lower on 28/28**
- **Drift guard `|C_T − C_1|`: +1.3 % mean** — this is the row that proves we are not
  simply freezing the texture, and it must be printed next to flicker.

Measured on texels (the PBR voxel field, `base_color` channels 0:3), not rendered
pixels, so rasterisation and shading cannot contribute.
Source: `out/texel_summary.json`, built by `render_rung27_orbit.py --texel-metrics`.

Keep tLP in the paper, demoted, reported as worse with the one-line reason. Better a
limitation we pre-empted than one a reviewer finds.

### B4 — Figure 3 contradicts the method text, twice

The system figure draws **Temporal Cross Attention as a "New layer" inside the 3D
Generation Model**, in the order Temporal → Spatial Cross-Attn → 3D Self-Attn.

1. **It is not a layer.** `blend_conds()` operates on the conditioning dict *before* the
   flow model runs. L313 says exactly this: "the one place temporal information can enter
   without touching the model is through the conditioning tokens." A figure showing it
   inside the block stack contradicts the paper's central claim — and that claim is what
   the parameter-free contribution rests on.
2. **The block order is reversed.** The real order is `self_attn → cross_attn → mlp`
   (`rung27_selfattn_lora.py:21`, citing `modulated.py:149-157`). The figure shows
   cross-attention first.

### B5 — L392–394. The target construction described is the version we abandoned

Paper: *"we copy the video's color at exactly the pixels our render occupies and leave
the rest background."*

`rung27_selfattn_lora.py:1682`:

```python
def inter_mask(fi):
    return renderer.mask & gt_masks[fi]
```

The **intersection** of our silhouette and the GT silhouette — used for the loss
(`:2298`) and for PSNR/SSIM (`:1703`).

This matters beyond wording: the render-silhouette version is what produced the white-patch
failure, where the silhouette-conflict band carried 84 % of the loss and drove sigmoid
saturation. Switching to the intersection took white vertices from 2.87 % to 0.00 %.
The paper is also internally inconsistent — L516 correctly says intersection for evaluation.

---

## C. In our favour — the window ablation already exists

Earlier notes (including `PAPER_STATUS_3DV27.md`) say a 1 / 3 / 5 window sweep does not
exist. **That is out of date.**

`mcfm_blend.py` implements `_OFFSETS['E'] = (-2,-1,0,1,2)`, and there are **24 `v2_E`
runs on disk** at rung27, 30 epochs. W = 1 is simply `mcfm=None`. **15 objects have all
three arms complete:**

| object | W=1 | W=3 | W=5 | best |
|---|---|---|---|---|
| ancient_lady_effect_2 | 28.801 | 28.743 | 28.794 | W=1 |
| animal_blob_crack | 25.144 | 25.275 | 25.297 | W=5 |
| animal_blob_orange_crack | 27.456 | 27.763 | 27.742 | W=3 |
| chair_ice | 21.370 | 21.562 | 21.462 | W=3 |
| chair_moss | 27.913 | 27.999 | 27.936 | W=3 |
| chair_real_wooden_crack | 25.573 | 25.310 | 25.319 | W=1 |
| dragon_mush | 24.682 | 24.632 | 24.644 | W=1 |
| dragon_mush2 | 23.262 | 23.286 | 23.279 | W=3 |
| fish_glitter | 23.531 | 23.691 | 23.739 | W=5 |
| fish_ink | 22.082 | 22.199 | 22.166 | W=3 |
| hand_rorschach | 24.946 | 25.057 | 24.956 | W=3 |
| napolean_teapot_crack | 24.481 | 24.489 | 24.546 | W=5 |
| napolean_waves | 22.083 | 22.339 | 22.369 | W=5 |
| octocat_clay | 27.435 | 27.453 | 27.466 | W=5 |
| octocat_shine | 27.242 | 27.285 | 27.318 | W=5 |

- mean Δ vs W=1: **W=3 +0.072 dB**, **W=5 +0.069 dB**
- better than W=1: W=3 on **12/15**, W=5 on **12/15**
- W=5 better than W=3 on **9/15** — a coin flip

This is a clean ablation that **justifies the default**: temporal context helps
consistently, and widening past three frames buys nothing. Table 2's
"Window 2-frame / 3-frame" row understates what we have — make it 1 / 3 / 5 over 15 objects.

Nine more objects (`spot_lava`, `pumpkin_rot`, the four octopus, both sheep,
`tie_fighter_bw`) have `v2_E` run dirs whose `final_eval.json` is unwritten; worth
checking whether those finished.

**Still true and important:** `rung32 --context-window` is a **different operator** —
neighbour frames concatenated into a longer key/value bank (1029 → 3087 tokens), no blend,
no gate, no extra adapter. `rung32_wide_context_lora.py` asserts it is mutually exclusive
with `--mcfm`, on the grounds that blending then concatenating "is neither method and
answers no question." **Do not merge the two into one ablation row.**

---

## D. Internal inconsistencies

| line | problem |
|---|---|
| Fig. 2 caption | says "skull … teapot"; the figure and L460–464 show a **chair** and a **plane**. Caption is stale from an earlier figure. |
| L464 vs Fig. 2 label | body says "ocean waves rolling in"; the figure label says "Japanese-style waves gushing" |
| L476 | `??` — broken `\ref`, renders literally |
| L484 | `tLP [? ]` — broken `\cite` |
| L486 vs Table 2 | text says tLP is reported in Tab. 2; Tab. 2's column is **Jitter**, and there is no tLP row |
| Baselines ¶ vs Table 1 | text names five methods (SV4D 2.0, L4GM, DreamGaussian4D, MeshNCA, Tex4D); **none of them appears** as a Table 1 row |
| L524 | "We report **four** quantities", then five are listed |
| Table 1 | three metric columns for a protocol promising four-plus |
| Figs. 8 and 9 | **identical figure, identical caption, printed twice** on p. 10 |
| L568 | "Table 2 shows the ablation against temporal, spatial, and 3D self attention" — Table 2 has no such rows |
| L567 vs Fig. 7 caption | text says Fig. 7 shows frozen-TRELLIS flicker; the caption says it is the noise/conditioning grid |
| Abstract pt. 6 | "a published dynamic-texturing method" (one); §4.5 promises five baselines |
| Table 1 | "+ fixed noise only" is an ablation and also appears in Table 2 |

**Prompts.** None of the figure-quoted prompts ("blue paint cracks and peels",
"Japanese-style waves gushing") appears in `data/dynamesh_meshes/KLING_PROMPTS.md` or
`data/_dale_fitted/KLING_PROMPTS_DALE.md`. On disk those effects are "wooden crack" and
the whale/plane clip. Display paraphrases are fine, but L446 promises the full prompt in
the supplementary, so the short-label → literal-prompt mapping has to be there.

---

## E. Claims that need care

### E1 — L369, "the choice of module matters far more than the choice of projection"

Supported, but not by the number you would reach for first. On `spot_lava`, no-MCFM,
30 epochs:

| arm | targets | PSNR |
|---|---|---|
| rung17 | kv | 22.670 |
| rung18 | qkv | 21.894 |
| rung25 | kvo | 22.193 |
| rung19 | qkvo | 22.670 |
| **rung27** | **qkvo+sa** | **24.178** |

Projection spread within cross-attention: **0.776 dB**. Adding the self-attention module:
**+1.508 dB** — roughly twice the entire projection spread.

**But +1.508 dB sits just under our own stated 1.55 dB run-to-run noise floor**, so do not
defend this with one object's margin. Defend it with consistency: rung19 → rung27 is
positive on **24/24 objects**.

**Check first:** rung17 (`kv`) and rung19 (`qkvo`) both report **22.670** on spot_lava,
identical to three decimals. Confirm those are two distinct runs before any projection
ablation goes into the paper.

### E2 — Table 2 bolds "decoder blocks" as the winning adapter placement

There is no decoder-block adapter arm in rungs 27–34; every target is an attention
projection of the flow model. If that row is real it comes from somewhere outside this
ladder; if it is aspirational, a bolded number is a promise we may not be able to keep.

### E3 — "Adapter rank TBD"

Rank arms exist only at rung17 (r1 / r2 / r4 / r16 / r32 on `all_kv`), a **different target
set** from rung27's. Justifying rank 4 from rung17 data needs saying so explicitly, or
re-running the sweep at rung27.

### E4 — L106–107 and L474, viewpoint coherence

*"coherent from viewpoints unseen in the driving video"* / *"the effect is present and
consistent on both, including from unseen views."*

Our own failure material shows the opposite for structured patterns: the vase's floral
medallions dissolve into smeared blobs by yaw 180° and the dot rows stop reading as rows;
the penguin's circuit traces blur and break. Both are in the supplementary reviewers will
watch. Scope the claim to diffuse / organic effects and let the failure figure carry the rest.

### E5 — L111–113, "the mesh and the adaptation never interact"

Too absolute. Fig. 4 shows transfer works, which is not the same as no interaction — the
self-attention adapter is fit on one object's voxel graph. "Are not coupled by
construction" survives review; "never interact" invites a counterexample.

### E6 — Fig. 7's "smoothness knob"

Check the provenance of whichever runs back it. The v4 regularisation was a **no-op** —
`lambda_reg` never took effect because of a tensor-view aliasing bug, so that arm ran
unregularised. Any smoothness ablation sourced from those runs measures nothing.

---

## F. Measured results the paper does not yet use

1. **The temporal result.** Flicker −20.0 % and jerk −44.4 %, both on **28/28** objects,
   with drift held to +1.3 %. Stronger than anything §4.2 currently claims, and already
   measured. This should be the headline.
2. **rung31 justifies "parameter-free".** A second gated cross-attention pass over the
   window with its own LoRA **ties** parameter-free MCFM (+0.019 dB, inside noise, 24
   objects). That is the answer to "why not just learn it?", which a reviewer will ask.
3. **KL closes a question.** 144 runs across 9 objects, **0/144 beat the β=0 control**,
   monotone in both weights. Two sentences of supplementary forecloses a reviewer suggestion.
4. **The comparison table is fully populated** — 24 objects × 7 methods, flicker /
   acceleration / drift at the supervised view and three unseen views, plus PSNR/SSIM, with
   sign tests. Scored as **distance from the driving clip's rate**, ours is first in all
   three columns (|ΔFlicker| 0.02098, |ΔAccel| 0.03411, |ΔDrift| 0.1013). Beats frozen
   22/24 (p = 3.6e−05) and SV4D 2.0 20/24 (p = 0.0015). L4GM is a sign-test tie (10/24
   closer, p = 0.54) but we lead on the mean, 0.02098 vs 0.02762. The claim to write is
   **"closest to the driving clip's rate"**, not "lowest flicker".

---

## G. Shape budget — check before Figure 5 is built

Under the rule *no same shape + same effect twice; a shape may carry at most two effects*:

- **spot** is full — lava (Fig. 3) and Rorschach (Fig. 4)
- **nefertiti** has the teaser
- **teapot** already appears in Fig. 4's Rorschach transfer
- **chair** has paint-cracks (Fig. 2) and lava (Fig. 4) — full

So a diversity panel using nefertiti ×3 or teapot ×3 breaks the rule. Worth settling before
that figure is drawn. Separately, the stale Fig. 2 caption would have put **skull + lava**
in the gallery while Fig. 4 already uses skull + lava — another reason it has to go.

---

## Priority

If only three things are fixed: **B3** (tLP), **B4** (Figure 3), **B5** (target
construction). Those are the three where a careful reviewer does not find a typo — they
find a claim our own data or our own code contradicts.
