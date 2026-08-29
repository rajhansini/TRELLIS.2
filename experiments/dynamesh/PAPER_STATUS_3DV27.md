# 3DV '27 — status of my items (Raj)

Last updated 2026-08-20. Every status below is what is ON DISK, not what is planned.
Numbers are training-view PSNR against the 2D-copy targets unless stated.

---

## Experiments

### Temporal attention

**Spatio-temporal attention (joint) — DONE, and it is a negative result.**
Pooling all W x N tokens into one softmax lets token j be answered by a different
token from a different frame, so the conditioning stops being spatially anchored.

| object | no MCFM | temporal only | joint | joint delta |
|---|---|---|---|---|
| pumpkin_rot | 33.723 | 33.816 | 32.744 | -0.979 |
| skull_lava | 25.347 | 25.624 | 24.038 | -1.309 |
| hand_rorschach | 24.831 | 25.057 | 23.597 | -1.234 |
| spot_lava | 23.994 | 24.194 | 22.659 | -1.335 |
| teapot_lava2 | 22.624 | 22.688 | 20.336 | -2.288 |

Loses on 5 of 5, and the margin clears the 1.55 dB run-to-run noise floor on four of
them. Rendered too -- 16 videos, 4 objects x (training view + 3 unseen).
Artifact: "Which Blend Wins".

**Temporal-then-spatial with separate LoRA for each — DONE (rung31), 4 objects.**
Second, gated cross-attention pass over the window with its OWN LoRA, zero-init gate
so the model is bit-identical to rung27 at initialisation.

| object | rung27 | rung31 (dual branch) |
|---|---|---|
| pumpkin_rot | 33.723 | 33.793 |
| skull_lava | 25.347 | 25.628 |
| hand_rorschach | 24.831 | 24.956 |
| spot_lava | 23.994 | 24.166 |

Gains are +0.07 to +0.28 dB -- inside noise. NOT a win on PSNR; needs the flicker
metric before any claim.

**A correction I owe the group.** I previously said we tested three ORDERINGS and
temporal-first won. The evidence does not support it. At the SLaT level phase7
favours temporal-first by 0.0064 dB, phase8 REVERSES and favours spatial-first by
2.12 dB, and phase8_gl puts all four variants within 0.000174 dB. Measured properly
at the conditioning-token level, temporal-only vs spatial-then-temporal is 2-2 across
four objects with margins +0.029 / -0.296 / -0.147 / +0.008 dB. **Do not put an
ordering claim in the paper.** What IS supported is factorized-beats-joint.

The honest reason to ship temporal-only is not that it wins: the frozen
cross-attention already performs the spatial pass, so an explicit one is redundant
work done before the model ever sees the tokens.

**Not yet run: rung32** -- context simply widened to 3 frames (1029 -> 3087 tokens),
no second pass, no gate, no extra adapter. Code ready and gated; zero jobs so far.

### KL divergence regularization

**DONE, negative result, 96 cells.** KL between the LoRA-updated attention
distribution and the frozen one, on cross (rung29), self (rung28) and both (rung30),
independently weighted.

Every one of the 96 cells scored BELOW the beta=0 control. The grid is monotone in
both weights: best is the weakest penalty on both axes (beta_c=15%, beta_s=15%,
24.718 dB), worst is the strongest (23.897 dB), spread 0.82 dB. Raising either weight
lowers the score at essentially every point, and the two penalties compound rather
than trading off.

So on this data the best available KL weight is zero. It is worth reporting as an
ablation that closes the question, not as a component.

Caveat to state if we report the percentages: beta is set from a 6-epoch probe, and
the KL/data ratio was still falling ~30%/epoch when the probe stopped. Every cell
shares the convention so the ORDERING holds, but the axis labels are not exact shares
of the converged loss.

**Just built, not yet run:** rung33 (dual branch + KL both) and rung34 (wide context
+ KL both).

---

## Figures

**Diversity — same shape, different effects. READY, exceeds the ask.**
Asked for 2 shapes x 2-3 effects. Have:

| mesh | effects |
|---|---|
| nefertiti | 6 -- ancient_lady, statue_clay, lady_crack, teaser, effect_1, effect_2 |
| teapot | 3 -- crack, fungi, golden_crack |
| alien | 2 -- glow, spots |
| armadillo | 2 -- monster_lava_2, monster_rainbow |

13 of 20 objects share a mesh with at least one other, so this is the majority of the
set rather than an anecdote. All rendered at 6 cameras.
Artifact: "Six Cameras" -- 20 objects, 120 videos.

**Ablation (supplementary).** Partly covered:
- lora on cross-attention only = rung19, have it
- + self-attention = rung27, have it on 20 objects
- + temporal = rung31, have it on 4 objects
- window size: only w2 vs w3 measured, on 2 objects (+0.133, +0.013 dB -- both inside
  noise). **A 1 vs 3 vs 5 sweep does not exist yet.** rung32/34 accept window 5, so
  this is cheap to add.

**Failure cases (supplementary).** Two candidates, both already characterised:
- `teapot_porcelain` -- frames 1-45 score 23.968 dB, frames 46-121 score 16.769 dB.
  The video model deforms the teapot, so ~21% of targets after frame 46 are invented
  rim fill. This is a data failure, not a method failure, and it is the honest one to
  show.
- `plane_waves` -- 17.941 dB, weakest object in the set. Whale mesh, 68k px
  silhouette, the smallest; less surface to carry the caustic pattern.

**Comparison to baselines — my part not started.** No baseline method has been run on
our inputs by me.

---

## Tables

### Published tables — artifact registry (2026-08-26)

Three artifacts, all over the SAME 24 objects (batches A + B + C, the roster in
`out/texel_r19_params.tsv`). The metric SPACE differs between them and the numbers do
NOT reconcile across spaces — that is expected, not a bug. Always state the space.

| Artifact | Space | Objects | URL |
|---|---|---|---|
| Dynamesh Ablations | **3D texel-space** | 24 (A+B+C) | https://claude.ai/code/artifact/9b57c181-bc5d-46ef-ae90-0ebf7eafb0d8 |
| Supplementary Ablations | **3D texel-space** | 24 (A+B+C) | https://claude.ai/code/artifact/12e9de5c-604b-46b0-bd2d-81e44f2bd9fb |
| Dynamesh Comparison Table | **2D pixel-space** | 24 (A+B+C) x 4 cameras | https://claude.ai/code/artifact/648154fd-8fb7-4f63-9ed9-2b8ae15ed915 |

**3D texel-space** = computed on the decoded PBR voxel field C_t before any renderer,
so it is view-independent and covers surface no evaluation camera sees. No renders are
involved: the texel jobs run `--skip-render` (ODE + decoder, no PNGs, no ffmpeg).
PSNR/SSIM in those tables are read from `final_eval.json` of the run each texel json
names — `['final']` for an adapted arm, `['frozen']` for a frozen one.

**2D pixel-space** = computed on rendered frames at 512 px through the 4 cameras
(train + diagA/B/C), per-method foreground masks.

What each artifact holds:

- **Dynamesh Ablations** — Table 1 component ablation (main paper); Table 2 temporal
  configuration ablation (supplement), 72/72 cells.
- **Supplementary Ablations** — Supp Table 1 adapters-first ladder; Supp Table 2
  temporal-first ladder. Both 24/24. Share endpoints by construction.
- **Dynamesh Comparison Table** — Table A: 7 methods on SV4D2's shared 21-instant clock.
  Table B: the 4 methods that natively run 150 frames, at full rate.

Generators: `jobs/build_component_ladder.py`, `jobs/build_config_ablation.py`,
`jobs/build_supp_tables.py`, `jobs/fullrate_table.py`, and
`baselines4d/eval_video_metrics.py`.

## Job status — 2026-08-26

Live snapshot of every fleet. Regenerate rather than trust: counts move.

| fleet | what it is | trained | measured | queue |
|---|---|---|---|---|
| `m3_*v2_F/v2_G` | **window ablation** W=7 / W=11, rung 27 + MCFM temporal-only, 24 objects | running | 0/48 | 38 running, 2 pending |
| `r37_*` | **rung 37** per-position temporal LoRA, 42 objects (A–E) | 13/42 | 0/42 | **8 HELD** |
| `dl_*` | **dailies** for batches D+E, 18 obj x 2 arms x 4 cameras | n/a | n/a | 1 running, 143 pending |

### rung 37 — HELD, and why

**21 of 42 jobs hit TIMEOUT at the 4-hour walltime.** Completed runs take **2:54–3:59**,
so 4h was never enough headroom: rung 37 runs the cross-attention twice per block plus a
per-position mixer, roughly 3–4x the cost of an m3 arm (~51 min median).

State: 13 COMPLETED, 21 TIMEOUT, 8 held (`scontrol hold`, job ids 2214883–2214890).
Nothing was cancelled and no checkpoints were lost.

**Before releasing:** raise `--time` in `jobs/rung37_perpos.sbatch` from `04:00:00` to
`08:00:00`. Then `scontrol release 2214883 ... 2214890` and re-run
`jobs/submit_rung37.sh` — it is idempotent and its completion gate requires
`epochs=30` *and* a matching `n_frames`, so it re-submits only the incomplete cells and
skips the 13 that finished. The trainer resumes from `lora_best.pt`, so a requeued
job does not restart from scratch.

### Batches D and E

Training is **54/54 complete** across r19, r27 and r27+MCFM for all 18 objects, but
**no texel measurements exist**, so neither paper table can include D or E yet. Renders
for the dailies are submitted (`dl_*`); the texel pass is separate and not submitted.

### Batches A, B, C — the two paper tables

Complete and unaffected by any of the above: r19, r27, r27mcfm all 24/24, plus the
configuration ablation at 72/72. See the artifact registry above.

### Results that need a decision before submission

1. **W=5 beats our W=3.** In the configuration ablation `v2_E` improves on `v2_D` in
   every column — lower flicker on 23/24 objects (p=3.0e-06), lower acceleration on
   24/24 (p=1.2e-07), +0.05 dB, at marginally HIGHER drift so it is not damping. The
   blend is parameter-free, so W=5 costs nothing. Either adopt it as the headline and
   regenerate the temporal row everywhere, or keep W=3 and say in the supplement that a
   wider window improves further. Presenting W=3 as best is not available.
2. **The two ladders disagree — the components are not additive.** Temporal is worth
   -18.9% flicker as the LAST step and -34.2% as the FIRST; cross-attention is -58.8%
   then -50.3%. They compete for the same coherence budget. Report one ordering, state
   that it is an ordering, and never quote a delta as the component's intrinsic value.
3. **The 21-instant clock is not neutral between methods.** It inflates our unseen
   flicker x4.96 against x3.26 for MeshNCA, because a stride of ~7 frames discards
   exactly the frame-to-frame smoothness MCFM supplies. Ours vs frozen reads -49.8% at
   full rate and only -40.0% at 21 instants. Table B exists for this reason.
4. **Self-attention buys no coherence, in either ordering.** +1.30 dB / +1.27 dB on
   24/24, but flicker 15/24 (p=0.31) forward and +0.9% (p=0.84) reversed. It is a
   fidelity component. Reporting that is what makes the temporal row's 24/24 credible.

### Superseded below

Everything from here to the end of this section predates the 24-object passes and is
kept only as history. The 8-object and 20-object numbers are NOT the reported ones.

**Flicker.** Measured on the earlier 8-object set: -38.5% at the supervised view,
-41.5% on unseen views, lower on 32 of 32 object x azimuth cells, while the first
difference stays within a few percent so the texture is still moving as much -- just
less raggedly. Raw numbers in `out/flicker_all.json`.

**PSNR / SSIM.** 20 objects, both arms, in `out/psnr_all.json`.

**Not yet measured: flicker for the 3-mode comparison and for rung31.** That is the
metric that would actually settle both, since PSNR structurally cannot see a temporal
prior -- it scores each frame against its own target independently. All the
fixed-camera renders needed already exist, so it is a CPU job, not a GPU one.

---

## What I would prioritise next

1. **Flicker on the 3 modes and on rung31.** Cheap, and it is the only metric that can
   support or kill the temporal claims. Everything needed is on disk.
2. **Window ablation 1 / 3 / 5.** Asked for explicitly, currently only 2 objects at
   two settings, both inside noise.
3. **rung32** -- the one mechanism with zero results.
4. Baselines -- not started on my side.

## What I would NOT do

- Claim anything about ordering (see the correction above).
- Present KL as a component. It lost 96/96.
- Quote MCFM by PSNR. Mean +0.10 dB is a non-result; the flicker number is the claim.

---

# Figure roster — which object, which effect, where it goes

Read from the two drafts: the teaser is nefertiti + "Clay patterns on pharaoh head"
(= `ancient_statue_clay`), the system figure is the cow + "Emerging lava cracks"
(= `spot_lava`). Everything below is picked from the 20 objects that have BOTH arms
trained and all six cameras rendered, so any of these can be swapped without new
compute.

| figure | object | mesh | effect | rung27 / +MCFM | status |
|---|---|---|---|---|---|
| **Teaser** | `ancient_statue_clay` | nefertiti | clay patterns | 25.648 / 25.697 | drafted (Guan), views to fix |
| **System overview** | `spot_lava` | spot | emerging lava cracks | 23.994 / 24.194 | drafted, refinements ongoing |
| **Mechanism illustration** | `spot_lava` | spot | lava cracks | — | keep the same object as the system figure so the reader carries one example through |
| **Comparison to baselines** #1 | `mushroom_glow` | mushroom | bioluminescence | 29.138 / 29.297 | best-scoring object; renders ready, baselines not run |
| **Comparison to baselines** #2 | `spot_lava` | spot | lava cracks | 23.994 / 24.194 | same object as the system figure |
| **Diversity** shape A | `nefertiti` | nefertiti | 6 effects available | see below | READY |
| **Diversity** shape B | `teapot` | teapot_ceramic_crack | 3 effects available | see below | READY |
| **Robustness** | lava on 4 shapes | spot / skull / armadillo / horse | same effect | see below | READY, low priority |
| **Ablation** | `spot_lava` (+ `skull_lava`) | spot / skull | lava | full rung ladder exists | table partly done |
| **Failure** #1 | `teapot_porcelain` | teapot | sprinkling water | 19.446 / 19.508 | READY — data failure |
| **Failure** #2 | `plane_waves` | whale | underwater caustics | 17.941 / 18.126 | READY — weakest object |

## Diversity — same shape, different effects

Asked for 2 shapes x 2-3 effects. Both rows are already rendered at six cameras.

| shape | effect | object | rung27 / +MCFM |
|---|---|---|---|
| **nefertiti** | effect 2 | `ancient_lady_effect_2` | 28.801 / 28.743 |
| nefertiti | teaser | `ancient_lady_teaser` | 27.748 / 27.649 |
| nefertiti | effect 1 | `ancient_lady_effect_1` | 26.557 / 26.548 |
| nefertiti | clay | `ancient_statue_clay` | 25.648 / 25.697 |
| nefertiti | crackle | `ancient_lady_crack` | 25.027 / 24.965 |
| nefertiti | gold leaf | `ancient_lady` | 23.492 / 23.493 |
| **teapot** | fungal growth | `teapot_fungi` | 27.257 / 27.281 |
| teapot | golden crackle | `teapot_golden_crack` | 27.004 / 26.921 |
| teapot | crackle glaze | `teapot_crack` | 26.508 / 26.419 |

If three effects per shape: nefertiti = effect_2 / clay / crackle (visually distinct,
and clay is already the teaser so the reader recognises the shape); teapot = fungi /
golden crackle / crackle glaze. Avoid `ancient_lady` (gold leaf) -- it is the one
object whose Kling clip spills glitter outside the silhouette, 3.41% uncovered and
135 frames instead of 150, so it is not matched to the others.

Two more shapes are available if wanted: **alien** (glow 27.130, spots 24.932) and
**armadillo** (molten lava 26.905, rainbow iridescence 24.376).

## Robustness — same effect, different shapes

Lava/cracks across four meshes, all rendered:

| object | mesh | rung27 / +MCFM |
|---|---|---|
| `monster_lava_2` | armadillo | 26.905 / 26.951 |
| `skull_lava` | skull | 25.347 / 25.685 |
| `spot_lava` | spot | 23.994 / 24.194 |
| `horse_metal` | horse | 24.849 / — |

Marked lower priority in the plan and I agree -- the diversity figure already carries
the generalisation point.

## Ablation — what exists per candidate object

`spot_lava` is the only object with the complete ladder, which is why it should be the
ablation example:

| arm | spot_lava | skull_lava |
|---|---|---|
| rung19 cross-attn only | 22.670 | — |
| rung27 + self-attn | 23.994 | 25.347 |
| rung27 + MCFM temporal_only | 24.194 | 25.624 |
| rung27 + MCFM spatial_then_temporal | 24.223 | 25.328 |
| rung27 + MCFM joint | 22.659 | 24.038 |
| rung31 dual branch | 24.166 | 25.628 |
| window w2 vs w3 | 24.061 / 24.194 | not run |

Gap against the plan: the asked-for **window sweep 1 / 3 / 5 does not exist** -- only
w2 vs w3, on 2 objects, both inside noise. And **flicker is not measured for any of
the mode arms or rung31**, which is the metric that would actually separate them.

## Objects NOT proposed for any figure, and why

`eagle_blackness` (23.058) and `monster_rainbow` (24.376) are fine but add nothing the
rows above do not already cover. `alien_spots` and `spot_raurshaw` are recent and
untested visually by me. The 11 new `_dale_fitted` objects (octopus, chair, blob,
napoleon) have videos uploaded but no targets or training yet.

---

# THE RULE (Itai, 2026-08-20) — and what it forces

> **No same shape + same effect more than once in the paper.**
> A shape may appear with **two different effects at most**.
> Example: spot lava (system) + spot Rorschach (generalization) = spot is now FULL.

## Violations in the current plan

1. **lava appears twice** — system figure (spot lava) and the gallery.
2. **skull lava appears twice** — generalization AND gallery. **Fix: the gallery drops
   skull lava and takes a different result.**

## Shape budget after the committed assignments

| shape | used | slots |
|---|---|---|
| spot | lava (system), rorschach (generalization) | **FULL — no spot anywhere else** |
| nefertiti | clay (teaser) | 1 free |
| skull | lava (generalization) | 1 free |
| alien | generalization | 1 free |
| armadillo | generalization | 1 free |
| teapot | generalization | 1 free |
| falconstatue, hand, horse, mushroom, whale | nothing | 2 free each |

## Generalization figure — as specified

**Column 1**

| row | content | source |
|---|---|---|
| 1 | spot Rorschach — video | `spot_raurshaw`, as in the artifact |
| 2 | spot Rorschach — 3D, side view (like the teapot) | `spot_raurshaw` diagA or diagC |
| 3 | alien, facing left | `alien_glow` 27.130 or `alien_spots` 24.932 — **which effect?** |
| 4 | armadillo as is | `monster_lava_2` 26.905 or `monster_rainbow` 24.376 — **which?** |
| 5 | teapot as is | fungi 27.257 / golden crackle 27.004 / crackle glaze 26.508 — **which?** |

**Column 2**

| row | content | source |
|---|---|---|
| 1 | skull lava — video | `skull_lava`, as in the artifact |
| 2 | skull lava — 3D, a DIFFERENT view from the video | `skull_lava` diagA / diagB / diagC |
| 3-5 | three good generalizations, three shapes not used yet | see below |

**The three unused shapes, best object each:**

| shape | object | effect | rung27 | note |
|---|---|---|---|---|
| mushroom | `mushroom_glow` | glow | 29.138 | best-scoring object in the whole set |
| horse | `horse_loki` | loki | 25.691 | |
| hand | `hand_rorschach` | rorschach | 24.831 | **effect clashes visually with spot Rorschach** |
| falconstatue | `eagle_blackness` | sooty | 23.058 | clean alternative to hand |
| whale | `plane_waves` | caustics | 17.941 | weakest object; reserved for the failure figure |

**My pick: mushroom_glow, horse_loki, eagle_blackness.** The rule as written is about
shape+effect PAIRS, so hand+rorschach is legal alongside spot+rorschach -- but two
Rorschachs in the same figure reads as a repeat to a viewer who is not tracking
meshes, and eagle_blackness costs nothing to use instead. `plane_waves` is excluded
because it is the failure example.

## Gallery — replacing skull lava

Skull's second slot is free, so either swap the SHAPE or swap the EFFECT:

- keep skull, change effect — no other skull effect has been generated yet
- **drop skull, use a shape with slots free** — `mushroom_glow` (if not spent on the
  generalization), `eagle_blackness`, `horse_loki`, or a second nefertiti/teapot/alien
  effect

Whatever is chosen for the generalization figure cannot also be the gallery
replacement -- that is the same violation moved one figure across.

## Open questions I need answered before locking this

1. **alien** — glow or spots?
2. **armadillo** — lava or rainbow? (lava collides with the system figure's effect on a
   different shape; legal under the rule, but it is the third lava in the paper)
3. **teapot** — fungi, golden crackle, or crackle glaze?
4. **gallery** — which result replaces skull lava, given it must not be one of the five
   generalization entries?

## Also noted

`humanoid_mike_wazowski`, `aliens_ufo`, `vehicles_tie_fighter` generalize less well
than the rest. Those are three of the 11 new `_dale_fitted` meshes -- none of them are
trained yet on my side, so that judgement is from Guan's renders, not mine.

---

# Consolidated arm coverage — figure objects and the 11 new ones

PSNR at the training view. `—` means NOT TRAINED, not "zero". Generated from
`runs/*/final_eval.json`, so it is what is on disk at the time of writing.

## Figure objects

| object | rung27 | temporal_only | spatial→temporal | joint | rung31 | rung32 | rung33 | rung34 | flicker |
|---|---|---|---|---|---|---|---|---|---|
| `ancient_lady_effect_2` | 28.801 | 28.743 | — | — | — | — | — | — | ✓ r27+mcfm |
| `spot_lava` | 24.178 | 24.194 | 24.223 | 22.659 | 24.166 | — | — | — | ✓ r27+mcfm |
| `pumpkin_rot` | 33.765 | 33.816 | 33.824 | 32.744 | 33.793 | — | — | — | ✓ r27+mcfm |
| `hand_rorschach` | 24.946 | 25.057 | 24.910 | 23.597 | 24.956 | — | — | — | ✓ r27+mcfm |

## The 11 new objects (uploaded 2026-08-21)

| object | mesh | effect | frames | targets | any arm trained |
|---|---|---|---|---|---|
| `animal_blob_crack` | animals_blub | crack | 150 | — | no |
| `animal_blob_orange_crack` | animals_blub | orange crack | 150 | — | no |
| `chair_real_wooden_crack` | furnature_chair | wooden crack | 150 | — | no |
| `chair_ice` | furnature_chair | ice | 150 | — | no |
| `chair_moss` | furnature_chair | moss | 150 | — | no |
| `napolean_teapot_crack` | statues_napoleon | teapot crack | 150 | — | no |
| `napolean_waves` | statues_napoleon | waves | 150 | — | no |
| `octopus_tar` | animals_octopus | tar | 150 | — | no |
| `octopus_sparkle` | animals_octopus | sparkle | 150 | — | no |
| `octopus_rainbow` | animals_octopus | rainbow | 150 | — | no |
| `octopus_rust` | animals_octopus | rust | 150 | — | no |

## What this table says

- **15 of 32 figure-object cells are empty.**
- `ancient_lady_effect_2` (the teaser shape) has only rung27 and temporal_only —
  it is missing spatial→temporal, joint and rung31, so it cannot appear in the
  ablation table as things stand.
- **rung32 / 33 / 34 have no trained run on any object.** The code is smoke-proven
  (2 epochs, mechanisms fire, correct rung stamped) but there are zero results.
- The 11 new objects have frames only. No mesh placed, no targets, no training.
- Filling the figure-object table = 3 arms for `ancient_lady_effect_2` + 12 for
  rung32/33/34 across the four objects = **15 training jobs**, then flicker.

---

# Writing — status per section

Merged in from PAPER_PLAN_3DV27_RAW.md on 2026-08-22, which is now deleted; this is
the single 3DV file.

| section | owner | status |
|---|---|---|
| Abstract | — | later |
| Introduction | — | draft DONE, pass pending |
| Related Work | — | DONE (draft, pass, comments, second pass all done) |
| Method | — | draft DONE |
| Experiments | — | draft pending |
| Limitations | Raj | examples + writeup pending; `teapot_ceramic_crack` named |
| Conclusion | — | later |

## Comparison to baselines — as circulated

- HIGH PRIORITY.
- Run baseline methods (multi-view video) on our input videos and compare.
- **MeshNCA** comparison — status ONGOING, not started on my side.
- At least two examples.

## Tables — as circulated

**Ablation.** One visual example plus one metric table for that example. Run the
window experiment at 1 / 3 / 5 frames. Metrics: flickering, and PSNR / SSIM against
the reference view.

**Comparison.** Report the flickering metric for baselines and for ours.

## Figure sections not otherwise covered above

- **Teaser** — fix views.
- **System overview** — visuals DONE, refinements ongoing.
- **Technical mechanism illustration** — explain the concept, draft the figure, then
  refinement iterations. All three TODO.
- **Robustness** (same effect, different shapes) — feels redundant given the
  generalization figure; lower priority.
- **Failure cases** (supplementary, lower priority) — structural texture failures,
  2 examples.

---

# Rung ladder

| rung | what it is |
|---|---|
| **27** | cross-attention + self-attention LoRA (baseline arm) |
| **28** | rung27 + KL on **self**-attention |
| **29** | rung27 + KL on **cross**-attention |
| **30** | rung27 + KL on **both** |
| **31** | dual gated branch — second cross-attention pass over the window, own LoRA + zero-init gate |
| **32** | wide context — cross-attention context widened 1029 → 3087 tokens, no second pass, no gate |
| **33** | rung31 + KL on both |
| **34** | rung32 + KL on both |

# Objects in play — mesh and effect

| object | mesh | effect |
|---|---|---|
| `ancient_lady_effect_2` | nefertiti | effect 2 |
| `spot_lava` | spot | emerging lava cracks |
| `pumpkin_rot` | pumpkin | rot |
| `hand_rorschach` | hand | rorschach |
| `animal_blob_crack` | animals_blub | crack |
| `animal_blob_orange_crack` | animals_blub | orange crack |
| `chair_real_wooden_crack` | furnature_chair | wooden crack |
| `chair_ice` | furnature_chair | ice |
| `chair_moss` | furnature_chair | moss |
| `napolean_teapot_crack` | statues_napoleon | teapot crack |
| `napolean_waves` | statues_napoleon | waves |
| `octopus_tar` | animals_octopus | tar |
| `octopus_sparkle` | animals_octopus | sparkle |
| `octopus_rainbow` | animals_octopus | rainbow |
| `octopus_rust` | animals_octopus | rust |

## Second new batch — 12 objects (uploaded 2026-08-22)

Mesh column is IoU-verified against the video's own first frame, not assumed.
All 12 will run the full ladder (runs 27–34, each with and without temporal-only
MCFM except 32/34, which take none) = 14 cells each, 168 cells.

| # | video file | object | mesh | IoU |
|---|---|---|---|---|
| 1 | `octocat_clay.mp4` | `octocat_clay` | thingi10k_octocat | 0.992 |
| 2 | `octocat_shine.mp4` | `octocat_shine` | thingi10k_octocat | 0.993 |
| 3 | `sheep_soil.mp4` | `sheep_soil` | thingi10k_wooly_sheep | 0.998 |
| 4 | `sheep_mud.mp4` | `sheep_mud` | thingi10k_wooly_sheep | 0.998 |
| 5 | `dragon_mush.mp4` | `dragon_mush` | dragons_xyzrgb_dragon | 0.992 |
| 6 | `dragon_mush2.mp4` | `dragon_mush2` | dragons_xyzrgb_dragon | 0.993 |
| 7 | `vehicles_tie_fighter_black_and _white.mp4` | `tie_fighter_bw` | vehicles_tie_fighter | 0.998 |
| 8 | `penguin_ice.mp4` | `penguin_ice` | penguin_circuits | 0.846 |
| 9 | `penguin_orange_mush.mp4` | `penguin_orange_mush` | penguin_circuits | 0.846 |
| 10 | `penguin_real.mp4` | `penguin_real` | penguin_circuits | 0.847 |
| 11 | `fish_glitter.mp4` | `fish_glitter` | animals_blub | 0.997 |
| 12 | `fish_ink.mp4` | `fish_ink` | animals_blub | 0.997 |

6 shapes: octocat ×2, sheep ×2, dragon ×2, tie_fighter ×1, penguin ×3, fish ×2.

**Caveat on the penguins.** Every other object matches its mesh at 0.992–0.998 with
the runner-up ~0.3 behind. The three penguins sit at 0.846 with the runner-up at
0.764 — a thin 0.08 margin. That is the profile of a near-miss mesh or a slightly
different pose. GATE-align (coverage, dies above 10% uncovered) decides; if it
fails, no targets are built and the watchdog will not retry a GATE failure.

**Mesh tree matters.** Each mesh is taken from the same directory as the render it
matched: `octocat` matched the base `renders/`, the other five shapes matched the
re-angled `kling/renders/`. Using the wrong tree bakes in a wrong pose that still
produces plausible-looking targets.

# Batch roster — A / B / C

| batch | object | video |
|---|---|---|
| **A** | `ancient_lady_effect_2` | ancient_lady_effect_2.mp4 |
| **A** | `spot_lava` | kling_spot_lava.mp4 |
| **A** | `pumpkin_rot` | kling_pumpkin_rot.mp4 |
| **A** | `hand_rorschach` | hand_rorschach.mp4 |
| **B** | `animal_blob_crack` | animal_blob_crack.mp4 |
| **B** | `animal_blob_orange_crack` | animal_blob_orange_crack.mp4 |
| **B** | `chair_real_wooden_crack` | chair_real_wooden_crack.mp4 |
| **B** | `chair_ice` | chair_ice.mp4 |
| **B** | `chair_moss` | chair_moss.mp4 |
| **B** | `napolean_teapot_crack` | napolean_teapot_crack.mp4 |
| **B** | `napolean_waves` | napolean_waves.mp4 |
| **B** | `octopus_tar` | octopus_tar.mp4 |
| **B** | `octopus_sparkle` | octopus_sparkle.mp4 |
| **B** | `octopus_rainbow` | octopus_rainbow.mp4 |
| **B** | `octopus_rust` | octopus_rust.mp4 |
| **C** | `octocat_clay` | octocat_clay.mp4 |
| **C** | `octocat_shine` | octocat_shine.mp4 |
| **C** | `sheep_soil` | sheep_soil.mp4 |
| **C** | `sheep_mud` | sheep_mud.mp4 |
| **C** | `dragon_mush` | dragon_mush.mp4 |
| **C** | `dragon_mush2` | dragon_mush2.mp4 |
| **C** | `tie_fighter_bw` | vehicles_tie_fighter_black_and _white.mp4 |
| **C** | `fish_glitter` | fish_glitter.mp4 |
| **C** | `fish_ink` | fish_ink.mp4 |
