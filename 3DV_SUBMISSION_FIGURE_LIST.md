# 3DV submission — figure list

Living checklist. One row per figure slot. `✅` = on disk and checked by eye,
`🔄` = running, `❌` = not started. Update the status column, not the plan.

The story every figure serves: **a temporal effect living on a fixed 3D shape.**
Geometry is provably unchanged across the sequence; only the texture moves. Every
figure has to make that legible without a caption.

---

## Paper names → asset on disk

Three different `teapot*crack` assets exist and they are NOT the same data. The paper
name is the disambiguator; the on-disk directory is what every `config.json` points at.

**Do not rename the directories.** Every run under `experiments/dynamesh/runs/` stores
absolute mesh and GT paths in its `config.json`; renaming orphans them. The paper name is
an alias recorded here and used in figure captions only.

| paper name | asset dir | frames | mesh |
|---|---|---|---|
| **`spot_lava_system_figure`** | `data/spot_lava` | 121 | `spot_lava_render_frame*` |
| **`teapot_crack_main_paper`** | `data/teapot_ceramic_crack` | 121 | `…_render_frame_guan.obj` |
| **`hand_rorschach_main_paper`** | `data/hand_rorschach` | — | — |
| **`skull_lava_main_paper`** | `data/skull_lava` | 150 | `skull_lava_render_frame.obj` |
| **`pumpkin_rot_main_paper`** | `data/pumpkin_rot` | 121 | `pumpkin_rot_render_frame_guan.obj` |

No results are recorded in this file. Numbers go in only after a run is submitted,
finished, and checked by eye.

Not the paper assets — kept only so nobody picks them up by mistake:

| asset dir | frames | why not |
|---|---|---|
| `data/teapot_ceramic_crack_correct` | 150 | separate capture, own mesh — **`_correct` in the name implies it supersedes the one above; unresolved, see Open** |
| `data/teapot_crack` | 150 | separate capture |

---

## Fig. 1 — System / key figure

**`spot_lava` is the system key figure. Decided, not a candidate.**

| paper name | asset dir | effect | status |
|---|---|---|---|
| `spot_lava_system_figure` | `data/spot_lava` | lava | locked in; poses per Guan — diagonal to one side, then diagonal again |

---

## Fig. 2 — Teaser  · **example NOT chosen yet**

Rotating video shown as frames. **The shape is re-posed AND the effect advances at the
same time**, so the reader cannot mistake it for a turntable of a static texture or for
a 2D video. 4–6 frames per row.

**Open: which asset carries the teaser. Nothing assigned.** Do not default to
`spot_lava_system_figure` — that slot is the system figure and reusing it wastes the teaser.

---

## Fig. 2b — Main-paper figures already named

| paper name | asset dir | effect | status |
|---|---|---|---|
| `teapot_crack_main_paper` | `data/teapot_ceramic_crack` | crackle glaze | also fills gallery slot 3.2 |
| `hand_rorschach_main_paper` | `data/hand_rorschach` | Rorschach ink | |
| `skull_lava_main_paper` | `data/skull_lava` | lava | also a candidate for Fig. 4 (same effect, different shape) |
| `pumpkin_rot_main_paper` | `data/pumpkin_rot` | rot | also fills gallery slot 3.4 |

---

## Fig. 3 — Gallery

Same claim, breadth. **3 views per asset**, 4–6 timesteps. Minimum 4 assets, **target 6**.

| # | asset | effect | GT video | status |
|---|---|---|---|---|
| 3.1 | `spot_lava_system_figure` | lava | ✅ | candidate |
| 3.2 | `teapot_crack_main_paper` | crackle glaze | ✅ | candidate |
| 3.3 | `whale_spots` | spots | ✅ | candidate |
| 3.4 | `pumpkin_rot_main_paper` | rot | ✅ | candidate |
| 3.5 | `penguin_circuits` | circuitry | ✅ | candidate |
| 3.6 | `horse_metal` | metal | ✅ | candidate |

Bench, promote if a slot opens: `ancient_lady` · `plane_waves`.

---

## Fig. 4 — Same effect, different shape  · **generalization**

**This is effect TRANSFER, not retraining.** The LoRA learned on
`spot_lava_system_figure` is applied unchanged to a different mesh. No GT video and no
training is needed for the target shapes — that is the whole point of the claim. Do not
substitute a run trained on `data/horse_lava` or `data/pumpkin_lava`; those are separate
captures and using them would prove nothing about transfer.

| effect | source (trained on) | target meshes | status |
|---|---|---|---|
| **lava** | `spot_lava_system_figure` | `data/horse_metal/mesh/horse_metal_render_frame_guan.obj` · `data/pumpkin_rot/mesh/pumpkin_rot_render_frame_guan.obj` | ❌ |
| **effect #2 — NOT CHOSEN** | ? | ? | ❌ **need one more transferable effect** |

Note for effect #2: it has to be an effect whose source asset is already trained, and it
should transfer to shapes that are visually different from the source — the weaker the
shape similarity, the stronger the claim.

Requires: the trained LoRA to be loadable against a mesh other than the one it was fitted
on. Confirm `render_rung27_orbit.py --mesh <other>.obj` accepts this before rendering —
if the checkpoint is tied to the source mesh's voxel layout, this figure needs new code.

---

## Fig. 5 — Same shape, different effect  · **2 figures** (generalization)

| # | shape | effects | status |
|---|---|---|---|
| 5.1 | teapot | `teapot_crack_main_paper` ✅ · `teapot_lava2` ✅ | ❌ — `teapot_porcelain` moved to Limitations |
| 5.2 | spot | `spot_lava_system_figure` ✅ · `spot_star` ✅ · `spot` ✅ | ❌ |

---

## Fig. 6 — Limitations  · **its own section**

| paper name | asset dir | effect | status |
|---|---|---|---|
| `teapot_porcelain_limitation` | `data/teapot_porcelain` | sprinkling water / wet sheen | ❌ |

Not a gallery or generalization asset — it carries the limitation section on its own.

---

## Supplementary — ablation

| item | status |
|---|---|
| rung ladder (kv → qkv → qkvo → qkvo+sa) | ❌ |
| LPIPS weight sweep {0, 0.05, 0.1, 0.2, 0.5} on rung27/horse_metal | ❌ |
| MCFM v2 vs v3 (temporal token blending) | ❌ |
| LoRA rank {1,2,4,8,16,32} | ❌ |

---

## Open — blocking, answer before rendering starts

1. **`alien` and `bsb`** — named in the figure brief but not in the 18-mesh set
   (`armadillo falconstatue hand horse_metal lionstatue mushroom nefertiti
   penguin_circuits pumpkin_rot skull spot spot_lava spot_star sword
   teapot_ceramic_crack teapot_porcelain whale_spots wingnut`). Which assets are these?
2. **`iseg`** — the iSeg paper's figure layout as a template, or the multi_iSeg meshes as
   an asset?
3. **Which teapot crack is the real one** — `teapot_ceramic_crack` (121 frames,
   Guan-fitted mesh) is currently `teapot_crack_main_paper`, but
   `teapot_ceramic_crack_correct` (150 frames, own mesh) is named as if it replaced it.
4. **Spot's poses** — "diagonally to one side, then diagonally again". Exact
   (yaw, elevation) pairs from Guan, or pick and have them check?

---

## Tooling

| need | tool | status |
|---|---|---|
| teaser rows (varying pose + varying t) | `render_rung27_orbit.py --turns 1`, sample 4–6 frames | ✅ exists |
| ablation strips (fixed training view, all 121 frames) | `render_rung27_orbit.py --turns 0` | ✅ exists |
| **gallery: 3 discrete views × 4–6 timesteps** | none — orbit script only does continuous turns | ❌ ~40 lines, new file, does not touch rung27 |
| hero stills for new Kling clips | `data/dynamesh_meshes/render_hero.py` | ✅ 18/18 correct |
| Kling prompts | `data/dynamesh_meshes/KLING_PROMPTS.md` | ✅ 18 written |

## Kling prompt rules (learned the hard way)

- Never name an emissive noun (`glow`, `bolt`, `arc`, `spark`, `flame`, `ember`, `aura`) —
  the model paints it into the air. Use a material word: etched, inlaid, oxidised,
  tattooed, stained, tinted.
- State the background **positively** ("everything outside the outline is solid white"),
  not as a list of exclusions.
- Add `NO WRITING OF ANY KIND` — Kling adds glyphs unprompted on metal and on anything
  Egyptian.
- Add `THE FORM MUST STAY READABLE` with an explicit floor ("at least half the surface
  stays bare"), or high-contrast effects flood the mesh into a flat silhouette.
- Add `THE PATTERN GROWS, IT DOES NOT SLIDE` when the effect should nucleate and expand
  rather than translate across the surface.
- Check frame 1 vs the last frame for scale drift before accepting any clip; Kling
  push-ins survive a casual look and break the fit.

---

## Batch D — supplementary meshes (added 2026-08-26)

11 Kling assets over 5 new meshes, sourced from other lab repos via
`experiments/dynamesh/MESH_CANDIDATES.md`. Conditioning renders are the max-exposed-area
front views from `supplementary_meshes/front/` (Kling camera: 1024 px, dist 2.6, fov 30).
Full registry and the upload/sort procedure: `data/_batch_d/BATCH_D.md`.

| asset dir | mesh path | source of mesh |
|---|---|---|
| `data/moai_silver` | `data/moai_silver/mesh/moai.obj` | `share/projects/vcf/QuadWild_300_Meshes/Organic/moai.obj` |
| `data/moai_animated` | `data/moai_animated/mesh/moai.obj` | as above |
| `data/gargoyle_spiral` | `data/gargoyle_spiral/mesh/gargoyle.obj` | `share/projects/vcf/QuadWild_300_Meshes/Organic/gargoyle.obj` |
| `data/gargoyle_effect_one` | `data/gargoyle_effect_one/mesh/gargoyle.obj` | as above |
| `data/airplane_blub` | `data/airplane_blub/mesh/aircraft.obj` | `share/projects/vcf/QuadWild_300_Meshes/Mechanical/aircraft.obj` |
| `data/airplane_red_cracks` | `data/airplane_red_cracks/mesh/aircraft.obj` | as above |
| `data/teddy_bleach` | `data/teddy_bleach/mesh/teddy.obj` | `ddecatur/latent-nerf/shapes/teddy.obj` |
| `data/teddy_fusion` | `data/teddy_fusion/mesh/teddy.obj` | as above |
| `data/goat_burnt` | `data/goat_burnt/mesh/goat.obj` | `itailang/mesh_feature_fields/meshes/goat.obj` |
| `data/goat_flower` | `data/goat_flower/mesh/goat.obj` | as above |
| `data/goat_clay` | `data/goat_clay/mesh/goat.obj` | as above |

Asset prefix is `airplane_*` while the mesh file is `aircraft.obj`; kept as-is, since
renaming a mesh orphans the absolute paths stored in every run's `config.json`.

Status: meshes and front renders on disk ✅ · videos ❌ (pending upload) ·
frames ❌ · watertight/UV check ❌

---

# Supplementary

Table structures for the supplement. Numbers are the 24 objects of batches A–C,
seed 42, texel-space metrics on the decoded base-colour field. Drift is a guard,
never a score — no best value is marked for it.

## S1 · Adapters first

Frozen → CA → CA+SA → CA+SA+Temporal. The order the method section introduces them in.

| # | Configuration | Rung / arm | PSNR ↑ | SSIM ↑ | Flicker ↓ | Accel. ↓ | Drift |
|---|---|---|---|---|---|---|---|
| 1 | Frozen TRELLIS.2 | — (no training) | 11.50 | 0.3363 | 0.02074 | 0.03102 | 0.3621 |
| 2 | + LoRA CA | rung19 | 23.47 | 0.7209 | 0.00854 | 0.01175 | 0.2288 |
| 3 | + LoRA CA + LoRA SA | rung27 | 24.76 | 0.7620 | 0.00844 | 0.01153 | 0.2247 |
| 4 | + LoRA CA + LoRA SA + Temporal | rung27 + MCFM v2_D | **24.81** | **0.7630** | **0.00684** | **0.00701** | 0.2246 |

## S2 · Temporal first

The same four components in the opposite order. Rows 1 and 4 are numerically identical
to S1's first and last rows by construction, so whatever differs between the ladders is
attribution, not measurement.

| # | Configuration | Rung / arm | PSNR ↑ | SSIM ↑ | Flicker ↓ | Accel. ↓ | Drift |
|---|---|---|---|---|---|---|---|
| 1 | Frozen TRELLIS.2 | — (no training) | 11.50 | 0.3363 | 0.02074 | 0.03102 | 0.3621 |
| 2 | + Temporal Attention | frozen + MCFM v2_D (no training) | — | — | — | — | — |
| 3 | + Temporal Attention + LoRA CA | rung19 + MCFM v2_D | — | — | — | — | — |
| 4 | + Temporal Attention + LoRA CA + LoRA SA | rung27 + MCFM v2_D | **24.81** | **0.7630** | **0.00684** | **0.00701** | 0.2246 |

Rows 2–3 blank: measured on 12 of 24 objects, rest queued. A half-set mean is not
comparable to the full means above and below it. On the 12 that have landed, row 2 gives
−33.1% flicker and −54.4% acceleration on 12/12 (p=0.00049), PSNR +0.0% (7/12, p=0.77).

## S3 · Component ablation

S1 restated by rung, with the on/off columns. Main-paper table.

| Configuration | Cross | Self | Temp. | PSNR ↑ | SSIM ↑ | Flicker ↓ | Accel. ↓ | Drift |
|---|---|---|---|---|---|---|---|---|
| Frozen TRELLIS.2 | ✗ | ✗ | ✗ | 11.50 | 0.3363 | 0.02074 | 0.03102 | 0.3621 |
| rung 19 — cross-attention | ✓ | ✗ | ✗ | 23.47 | 0.7209 | 0.00854 | 0.01175 | 0.2288 |
| rung 27 — + self-attention | ✓ | ✓ | ✗ | 24.76 | 0.7620 | 0.00844 | 0.01153 | 0.2247 |
| rung 27 + MCFM v2_D — ours | ✓ | ✓ | ✓ | **24.81** | **0.7630** | **0.00684** | **0.00701** | 0.2246 |

Per step: cross +11.97 dB / −58.8% flicker (24/24) · self +1.30 dB (24/24) but flicker
not significant (15/24, p=0.31) · temporal +0.05 dB but −18.9% flicker, −39.2% accel (24/24).

## S4 · Configuration ablation

Every component on, rung 27 backbone fixed, only the blend configuration varied.
Two axes cross at v2_D: window width at fixed flavour, and flavour at fixed W=3.

| Temporal configuration | PSNR ↑ | SSIM ↑ | Flicker ↓ | Accel. ↓ | Drift |
|---|---|---|---|---|---|
| No blend — reference | 24.76 | 0.7620 | 0.00844 | 0.01153 | 0.2247 |
| Temporal only, W=3 v2_D — ours | 24.81 | 0.7630 | 0.00684 | 0.00701 | 0.2246 |
| Temporal only, W=5 v2_E | **24.87** | **0.7646** | 0.00643 | 0.00610 | 0.2258 |
| Spatial → temporal, W=3 st_D | 24.79 | 0.7625 | 0.00681 | 0.00686 | 0.2260 |
| Joint spatio-temporal, W=3 v3_D | 23.71 | 0.7298 | **0.00575** | **0.00568** | 0.2156 |

OPEN DECISION: W=5 beats our chosen W=3 in every column — lower flicker on 23/24
(p=3.0e−06), lower acceleration on 24/24 (p=1.2e−07) — at marginally *higher* drift
(0.2258 vs 0.2246), so it is not damping, and the blend is parameter-free so W=5 costs
nothing. Either adopt W=5 as headline and regenerate S1/S3's last row and both comparison
tables, or keep W=3 and state in the supplement that a wider window improves further.
Presenting W=3 as the best setting is not available.

## Runs required per object to fill S1–S4

7 training runs. Both frozen rows come free from the `frozen_*` fields of the r19 passes —
there are no separate frozen jobs.

| # | Arm (run-dir identity) | Fills |
|---|---|---|
| 1 | `rung19_l1_lp_all_qkvo_r4` | S1 r2, S3 r2 · + frozen row |
| 2 | `rung19_l1_lp_mcfmv2_D_all_qkvo_r4` | S2 r3 · + frozen+MCFM row (S2 r2) |
| 3 | `rung27_l1_lp_all_qkvo+sa_r4` | S1 r3, S3 r3, S4 "no blend" |
| 4 | `rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4` | S1 r4, S2 r4, S3 r4, S4 "ours" |
| 5 | `rung27_l1_lp_mcfmv2_E_all_qkvo+sa_r4` | S4 W=5 |
| 6 | `rung27_l1_lp_mcfmst_D_all_qkvo+sa_r4` | S4 spatial→temporal |
| 7 | `rung27_l1_lp_mcfmv3_D_all_qkvo+sa_r4` | S4 joint |

MCFM is a TRAINING-time flag, not an inference switch: each blend variant is its own run
dir with its own LoRA weights, so 4–7 cannot be collapsed into one rung27 run evaluated
four ways. Each run then needs its `texdep_*` job to write `out/TEXEL/<obj>_<arm>.json`;
PSNR/SSIM come from that run's `final_eval.json`.

---

## Batch E — supplementary meshes, second round (added 2026-08-26)

6 Kling assets over 3 meshes, all already used in batch D, so the meshes are shared and
only the effect differs. Conditioning renders are the same max-exposed-area front views
from `supplementary_meshes/front/`. Registry: `data/_batch_e/batch_e.json`.

| asset dir | mesh | mesh path | effect | align IoU |
|---|---|---|---|---|
| `data/ivysaur_petal` | ivysaur | `data/ivysaur_petal/mesh/ivysaur.obj` | falling petals | 0.9105 |
| `data/ivysaur_petal_2` | ivysaur | `data/ivysaur_petal_2/mesh/ivysaur.obj` | falling petals, v2 | 0.9086 |
| `data/blub_raurshaw` | blub | `data/blub_raurshaw/mesh/blub.obj` | Rorschach, sliding blots | 0.9142 |
| `data/blub_drying` | blub | `data/blub_drying/mesh/blub.obj` | terracotta drying, thin-first | 0.9139 |
| `data/pegasus_shine` | pegaso | `data/pegasus_shine/mesh/pegaso.obj` | burnish / shine | 0.8838 |
| `data/pegaso_effect_1` | pegaso | `data/pegaso_effect_1/mesh/pegaso.obj` | effect 1 | 0.8840 |
| `data/mosaic_painting` | moai | `data/mosaic_painting/mesh/moai.obj` | mosaic, contour rows first | 0.9322 |

Mesh sources: `ivysaur` from `itailang/instant-edit/data/ivysaur.obj`, `blub` from
`ddecatur/latent-nerf/shapes/blub.obj`, `pegaso` from
`share/projects/vcf/QuadWild_300_Meshes/Organic/pegaso.obj`.

`pegasus_shine` keeps that spelling while its mesh is `pegaso.obj`; not renamed, since a
mesh rename orphans the absolute paths stored in every run's `config.json`.

Orientation: `out/orient_batch_e/orientation.json`. spot_lava control returned 0.9913
unchanged, and each mesh's two assets agree with one another (ivysaur 0.9105/0.9086,
blub 0.9142/0.9139, pegaso 0.8838/0.8840) — an independent consistency check, since the
two were solved separately.

`mosaic_painting` is a seventh asset on the moai mesh, added after the first six. Its
solve returned 0.9322 against batch D's moai at 0.9320/0.9324 — a three-decimal match on
an independent solve, which is what confirms it was rendered on that mesh.

Status: videos ✅ · 150 frames @ 1440² ✅ · orientation ✅ · 2D-copy targets ✅ (150 each)
· training ❌. For the two paper tables only 3 arms are needed per asset — rung19,
rung27, rung27+MCFM v2_D — so batch E is 21 runs, not 49; frozen comes free from the
r19 pass and the comparison table's "ours" row is the same rung27+MCFM cell.

---

## Textured continuation — round 2 (added 2026-08-26)

**The idea.** Every asset so far was generated by handing Kling a **grey render** and a
prompt, so the video is a *birth* sequence: bare mesh, then pattern appears, then pattern
covers. The fit therefore has to model an appearance transient on top of whatever motion
is present. Round 2 removes the transient. We render our own rung27 + MCFM v2_D result at
the training view, take the **last frame**, and hand *that* back to Kling as the start
frame. The new video is texture already present and moving inside a fixed silhouette,
which is what "dynamic texture" actually means.

**Why this is worth a slot.** It is the one experiment that separates the two claims the
paper conflates. A birth video rewards a method for getting the texture *there*; a
continuation video only rewards it for keeping the texture *coherent while it moves*. It
also closes the loop: round 2's last frame can seed round 3, so sequence length stops
being capped by Kling's clip length.

**Start frames.** `experiments/dynamesh/out/KLING_START/<obj>_r27_mcfmv2_D_last.png`, cut
by `experiments/dynamesh/jobs/make_kling_start.py`, which asserts the `[frozen | ours]`
panel arithmetic rather than assuming it. Five exist: `hand_rorschach`, `pumpkin_rot`,
`chair_real_wooden_crack`, `plane_waves`, `napolean_waves`.

**Prompts.** `data/KLING_PROMPTS_CONTINUATION.md`. Four of the six DALE locks carry over
verbatim; two were written for a grey start and had to be replaced, and four are new. The
load-bearing ones:

| lock | why it exists |
|---|---|
| THE SURFACE IS ALREADY FINISHED | stops Kling rewinding to a clean object and replaying the birth sequence |
| MATERIAL ENTERS AND LEAVES | holds coverage constant *while* everything moves. The first draft held coverage by making patches trade in place, which produced pulsation — near-zero displacement, i.e. exactly the degenerate case the drift guard exists to catch |
| THE PATTERN TRAVELS | gives motion a path, a direction and a **countable rate** (crests per clip). "Travels a real distance" is unfalsifiable; a crest count is not |
| THE LIGHTING DOES NOT TRAVEL | once fast directional motion is demanded, the cheapest way for a video model to deliver it is to slide the whole image, which reads as the object turning and fails GATE-align |

**Pipeline.** `experiments/dynamesh/jobs/submit_continuation.sh`, `OBJ=<asset>`:

1. `jobs/targets_hero.sbatch` — 2D copy of the video onto our mesh, carrying GATE-align.
2. `jobs/fig27.sbatch` with `MODE=v2_D` — rung27 (qkvo+sa, rank 4, 30 epochs, seed 42),
   submitted `--dependency=afterok` on step 1.

The dependency is the point, not convenience: a continuation video whose object drifted
off our mesh must not silently train. GATE-align kills step 1 and step 2 never starts.

**Asset convention.** A continuation asset is a *new* asset dir, never an overwrite of
round 1, so round 1 stays reproducible. Its mesh is a byte-identical copy of the round-1
mesh — same geometry, new texture video — which is what makes the two rounds comparable.

### Cells

| asset | source | start frame | video | frames | submitted |
|---|---|---|---|---|---|
| `plane_waves_from_frame_150` | `plane_waves` | frame 150 of `view_plane_waves_mcfm_train` | `plane_waves_from_frame_150.mp4` | 150 @ 1440² | 2026-08-26 |
| `chair_real_wooden_crack_from_frame_150` | `chair_real_wooden_crack` | frame 150 of `fixview_chair_real_wooden_crack_mcfm_yaw0` | `chair_real_wooden_crack_from_frame_150.mp4` | 150 @ 1440² | 2026-08-26 |

The chair's continuation video arrived named `chair_creek_from_frame_150.mp4` and was
renamed on upload. The pipeline resolves mesh, targets and run directory from the asset
name, so the asset must carry the source asset's name or traceability back to round 1 is
lost.

### Job tracking

Submitted by `OBJ=plane_waves_from_frame_150 bash experiments/dynamesh/jobs/submit_continuation.sh`.
Both prefixes are in `watchdog.py`'s KIND table (`tgt_`, `27_`), both are in
`out/job_manifest.json`, both have rows in `JOBLOG.tsv`.

**plane_waves_from_frame_150**

| # | job | id | name | log | pass condition |
|---|---|---|---|---|---|
| 1 | 2D copy + GATE-align | `2218448` | `tgt_plane_waves_from_frame_150` | `out/tgt_2218448.log` | `out/gt_targets_plane_waves_from_frame_150/gt_targets.json` exists **and** `[GATE-align]` present without `FAILED` |
| 2 | rung27 + MCFM v2_D | `2218449` | `27_v2_D_plane_waves_from_frame_150` | `out/FIGRUNS/27_v2_D_plane_waves_from_frame_150_2218449.log` | log has `[FINAL] rung` **and** the run's `final_eval.json` exists |

**chair_real_wooden_crack_from_frame_150**

| # | job | id | name | log | pass condition |
|---|---|---|---|---|---|
| 1 | 2D copy + GATE-align | `2218622` | `tgt_chair_real_wooden_crack_from_frame_150` | `out/tgt_2218622.log` | `out/gt_targets_chair_real_wooden_crack_from_frame_150/gt_targets.json` exists **and** `[GATE-align]` present without `FAILED` |
| 2 | rung27 + MCFM v2_D | `2218623` | `27_v2_D_chair_real_wooden_crack_from_frame_150` | `out/FIGRUNS/27_v2_D_chair_real_wooden_crack_from_frame_150_2218623.log` | log has `[FINAL] rung` **and** the run's `final_eval.json` exists |

### GATE-align results

The column that decides whether the continuation premise holds. Round 1 is the grey-start
video on the same mesh, round 2 the textured-start continuation.

| asset | uncovered % | align IoU % | IoU min over 150 fr | white px inside | verdict |
|---|---|---|---|---|---|
| `plane_waves` (round 1) | 0.00 | 98.68 | 0.9703 | 0 | — |
| `plane_waves_from_frame_150` | **0.02** | **97.33** | **0.9548** | **0** | PASS |
| `chair_real_wooden_crack_from_frame_150` | pending | pending | pending | pending | pending |

The chair is a **bad pass**. It cleared only because the limit is 10%, and that limit was
calibrated on grey-start videos that land at ~0.00%. Kling redrew the chair ~11% larger in
frame 1 (241,756 video px against our 217,868) and then held it — the IoU spread over 150
frames is 0.0016, so this is a frame-1 registration offset, not temporal drift. Uncovered
pixels are not left white; `make_gt_targets.py` fills them from the nearest valid
neighbour, so 7.2% of that object's targets are nearest-neighbour fabrication forming a
coherent rim around the whole chair. `jobs/fit_align.sbatch` (scale × dy sweep, idempotent,
reads the pristine `.prefit` mesh) is the standard repair for exactly this.

### Result — plane_waves_from_frame_150

Both rows are rung27 qkvo+sa rank 4 + MCFM v2_D, 30 epochs, seed 42, same mesh, same
schedule. Only the driving video differs.

| video | frozen PSNR / SSIM | final PSNR / SSIM | gain | best epoch |
|---|---|---|---|---|
| round 1 — grey start (birth) | 9.767 / 0.3339 | 18.126 / 0.5819 | +8.360 dB | 28 |
| round 2 — textured start (continuation) | 9.413 / 0.3645 | **20.625 / 0.7420** | **+11.212 dB** | 28 |

The continuation fits **+2.50 dB and +0.160 SSIM better**, and its gain over frozen is
2.85 dB larger. That is the direction the idea predicted: with the grey-to-textured
transient removed, the adapter only has to model motion rather than motion plus onset.

**Do not report this as a result yet.** It is one object, one seed, and the two rows are
scored against *different targets* — each video's own frames. A continuation clip whose
texture merely moves may simply have lower total variation than a birth clip that goes
from bare mesh to full cover, in which case the higher PSNR reflects an easier target
rather than a better fit. Before this claim goes anywhere it needs (a) more objects, and
(b) the GT sequences' own temporal statistics — flicker and drift computed on the target
frames themselves — so the difficulty of the two target sets can be compared directly.

For the plane, Kling held the silhouette across 150 continuation frames. The 1.35-point
IoU drop is real but small, and `strict_px_min` went the other way — 67,909/68,412 in
round 1 against 68,392/68,412 in round 2, i.e. the continuation video covers our mesh
*more* consistently frame to frame. `white_px_inside: 0` means no unsupervised pixels
entered the loss.

Job 2 is `--dependency=afterok:2218448`, so it shows `PENDING (Dependency)` until the 2D
copy succeeds and never runs at all if GATE-align kills job 1.

To track:

```bash
# both jobs at a glance
squeue -j 2218448,2218449,2218622,2218623 -o "%.10i %.46j %.10T %.8M %R"

# the number that decides whether the continuation idea works
grep "GATE-align" experiments/dynamesh/out/tgt_2218448.log experiments/dynamesh/out/tgt_2218622.log

# training progress
tail -f experiments/dynamesh/out/FIGRUNS/27_v2_D_plane_waves_from_frame_150_2218449.log

# whole-log view with state refreshed from sacct
bash /net/projects/ranalab/rajhansini/joblog.sh | grep _from_frame_150
```

**The number that matters is `uncovered %` on the `[GATE-align]` line of job 1.** Round 1
started from a grey render whose silhouette we controlled; this video started from our own
render and was continued by Kling, so `uncovered %` measures whether Kling held the
silhouette across 150 new frames. Near zero means the continuation premise holds. A large
value means Kling moved the plane, and per the align-gate calibration note the CPU raster
must be checked with `flip_y=False` and the value validated against the recorded IoUs
before the video itself is blamed.

`fig27.sbatch` requests 4h and 30 epochs takes 3:33–3:52, so a TIMEOUT on job 2 is normal
and expected; `--requeue` plus resume-from-checkpoint covers it.

**Caveat on `plane_waves`.** It is a bench asset, not one of the paper's 24 (batches
A+B+C) — see line 86 above. The teaser figure already shows it, so the continuation
result inherits that same status: usable as a figure, not as a table row, unless the
asset is promoted.

---

# Comparison tables — batch D+E and all batches (added 2026-08-27)

NEW tables. `comparison_table.tex` and `fullrate_table.tex` (batches A--C, 24 objects)
are UNTOUCHED and still rebuild exactly as before.

## The PSNR reference changed, and it has to be stated

The published A--C table scores fidelity against the RAW video frame. That frame carries
the VIDEO's silhouette while our render carries OURS; they agree to IoU 0.89--0.93, so
about 8% of pixels disagree in a thin boundary sliver. On A--C's low-contrast effects
that costs ~1 dB. On D+E's high-contrast effects (black-and-white marble, cracked clay,
spiral ink) the same sliver costs 7--14 dB:

| object (ours, supervised view) | vs raw frame | vs 2D copy | delta |
|---|---|---|---|
| chair_ice (A--C) | 20.97 | 22.33 | +1.36 |
| moai_silver | 13.24 | 20.13 | +6.89 |
| goat_clay | 15.16 | 23.91 | +8.75 |
| moai_animated | 15.39 | 26.39 | +11.00 |
| teddy_fusion | 14.76 | 26.48 | +11.72 |
| gargoyle_spiral | 11.00 | 24.63 | +13.63 |

That is misregistration being scored as texture error. These tables therefore score
fidelity against the **2D copy** — the same video colour resampled into our silhouette,
which removes the boundary term and nothing else, and is what the trainer's own
`final_eval` has always used. Temporal columns are unaffected either way: flicker and
acceleration compare consecutive frames within one sequence, so a fixed boundary offset
cancels. Raw-frame numbers stay in `out/FULLRATE/`; 2D-copy numbers in `out/FULLRATE_CG/`.

## D+E only — 18 objects, full rate

| Method | sFlick ↓ | sAccel ↓ | sDrift | uFlick ↓ | uAccel ↓ | uDrift | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|---|---|
| GT — driving clip | 0.00633 | 0.00824 | 0.2130 | 0.00633 | 0.00824 | 0.2130 | — | — |
| Frozen TRELLIS.2 | 0.01430 | 0.02112 | 0.3148 | 0.01514 | 0.02249 | 0.3114 | 12.88 | 0.3145 |
| **Ours** | **0.00477** | **0.00518** | 0.2026 | **0.00440** | **0.00512** | 0.1749 | **25.12** | **0.7669** |
| MeshNCA | 0.02030 | 0.03603 | 0.2709 | 0.02105 | 0.03742 | 0.2779 | 8.08 | 0.1178 |
| L4GM | 0.00754 | 0.01100 | 0.1957 | 0.00594 | 0.00968 | 0.1618 | 21.22 | 0.6916 |

Sign test over the 18: vs frozen **18/18 on all four** temporal columns (p=7.6e-06);
vs L4GM 18/18 on supervised flicker and acceleration and on unseen acceleration,
15/18 on unseen flicker (p=0.0075); vs MeshNCA 15--17/18 (p<=0.0075).

## All batches — 42 objects, full rate

| Method | sFlick ↓ | sAccel ↓ | sDrift | uFlick ↓ | uAccel ↓ | uDrift | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|---|---|
| GT — driving clip | 0.00927 | 0.01185 | 0.2738 | 0.00927 | 0.01185 | 0.2738 | — | — |
| Frozen TRELLIS.2 | 0.01751 | 0.02618 | 0.3380 | 0.01818 | 0.02720 | 0.3362 | 12.14 | 0.3035 |
| **Ours** | **0.00736** | **0.00787** | 0.2650 | **0.00572** | **0.00641** | 0.1999 | **24.90** | **0.7978** |
| MeshNCA | 0.01756 | 0.03086 | 0.2565 | 0.01845 | 0.03260 | 0.2602 | 7.67 | 0.1054 |
| L4GM | 0.00998 | 0.01400 | 0.2504 | 0.00703 | 0.01105 | 0.1827 | 21.63 | 0.7280 |

Sign test over the 42: vs frozen **42/42 on all four** (p=4.5e-13); vs L4GM 40--41/42 on
three columns (p<=4.1e-10) and 32/42 on unseen flicker (p=0.00094); vs MeshNCA 32--35/42
(p<=0.00094). All twelve comparisons are significant at 0.001. On A--C alone, three of
them were not: L4GM unseen flicker 17/24 (p=0.064), MeshNCA supervised flicker 17/24
(p=0.064), MeshNCA acceleration 18/24 (p=0.023). The extra 18 objects close them.

## A second correction: the frozen row was frozen+MCFM

The published Table B reads the frozen arm out of the LEFT half of the `--mcfm` run
(`view_<obj>_27m_*`). Inside an `--mcfm` run the frozen arm receives the blended
conditioning too, so that panel is frozen+MCFM, not frozen: it inherits the temporal
smoothing the ablation is supposed to isolate. The correct frozen is the left half of
the non-MCFM run (`view_<obj>_27_*`). `fullrate_metrics.py` was fixed on 2026-08-26 and
every number in FULLRATE / FULLRATE_CG carries the fix.

The frozen baseline is therefore substantially JITTERIER than published (A--C supervised
flicker 0.01303 -> 0.01992), which strengthens the ablation rather than weakening it:
ours now beats frozen 24/24 on all four temporal columns for A--C, where the published
table read 21/24, 21/24, 24/24, 23/24. Ours, MeshNCA and L4GM are unaffected -- their
rows reproduce the published values to 3 decimals.

A--C recomputed on the corrected pipeline, for a like-for-like check: ours 0.00931 /
0.00988 / 0.00671 / 0.00739, 24.73 dB / 0.8210; frozen 0.01992 / 0.02997 / 0.02046 /
0.03074, 11.59 / 0.2952; L4GM 21.94 / 0.7553; MeshNCA 7.36 / 0.0961.

Baselines on D+E were generated from scratch: MeshNCA 18 fits + renders, L4GM 18,
SV4D2 18, DG4D 18 (plys + our-camera renders). SV4D2 and DG4D feed the 21-instant
table only and are not in the full-rate tables above.

Sources: `out/FULLRATE_CG/<obj>.json` via `jobs/fullrate_metrics_copygt.py`, aggregated
by `jobs/fullrate_table_scoped.py --dir FULLRATE_CG --objs <list> --label <name>` into
`out/fullrate_table_batchDE.json` / `_batchALL.json` / `_batchABC.json`.

---

# Supplementary — artifact index (added 2026-08-29)

Every published artifact, with what it is. The rule that splits the list: the main
paper carries **Table 1** (the baseline comparison) and **Figs. 1–9**. Everything
else is supplementary.

Artifacts are private until shared from the page's share menu. `/artifacts` in the
Claude Code terminal lists them; the gallery is at claude.ai/code/artifacts.

## Tables and ablations — supplementary

| artifact | one-liner |
|---|---|
| [Window Sweep Table](https://claude.ai/code/artifact/47159615-2807-4e7b-9764-15538b6c7a0c) | **Table 1 source.** All baselines plus ours at W=3/5/7/11, 42 objects, one reference. Baselines and the W=3 row reproduce `fullrate_table_batchALL.json` bit-for-bit. |
| [Component Ablation](https://claude.ai/code/artifact/93443f39-82c7-4a34-a721-82cfc6722195) | The 4-row ladder at W=11 (frozen → +cross-attn → +self-attn → +temporal), plus where the temporal row lands at each window width. 24 objects, texel-space. |
| [Temporal Configuration](https://claude.ai/code/artifact/a71a121a-0dc5-4b39-bd11-4c0e6d943cdb) | Two axes crossing at the shipped setting: window width (W=1/3/5/7/11) and blend flavour (temporal-only / spatial→temporal / joint). Shows joint costs 1.08 dB to tie us on flicker. |
| [Window Ablation](https://claude.ai/code/artifact/65086efd-f633-44e4-a564-35e8441890e7) | Texel-space window sweep W=1/3/5/7/11 over 42 objects, with the `window_ablation.tex` block. |
| [Learned vs Parameter-Free ⚖️](https://claude.ai/code/artifact/d4857454-ff5a-4dea-a792-ef465eaffe3f) | Rung 37 (learned per-position temporal LoRA) against parameter-free MCFM. Newer of the two. |
| [Learned vs Parameter-Free ⚔️](https://claude.ai/code/artifact/8d124889-8fcc-405e-ba52-44eea8e4597d) | Earlier version of the same comparison, kept for history. |
| [Supplementary Ablations](https://claude.ai/code/artifact/12e9de5c-604b-46b0-bd2d-81e44f2bd9fb) | Earlier supplementary ablation collection (S1–S4 territory). |
| [Dynamesh Ablations](https://claude.ai/code/artifact/9b57c181-bc5d-46ef-ae90-0ebf7eafb0d8) | Earlier rung-ladder ablation page. |
| [42-Object Comparison](https://claude.ai/code/artifact/ec05e3f8-0476-441d-b314-e976ae846d06) | The published 42-object full-rate comparison, on the 2D-copy reference. |
| [Dynamesh Comparison Table](https://claude.ai/code/artifact/648154fd-8fb7-4f63-9ed9-2b8ae15ed915) | Earlier comparison table, batch D+E era. |
| [Window 11 Comparison](https://claude.ai/code/artifact/3cb51284-bce8-4a76-991d-56f01a0166c6) | **Superseded — do not cite.** Built on the raw-frame reference, so its PSNR/SSIM read ~6 dB low (ours 19.07, L4GM 22.20). Its temporal columns are correct and identical to the current table. |

## Figure drafts — main paper

| artifact | one-liner |
|---|---|
| [Duck Figure Draft](https://claude.ai/code/artifact/653ae2b0-de4b-4ad2-9be6-892d2fc6a176) | Fig. 7 textured-input figure, later draft. |
| [Duck Figure Draft](https://claude.ai/code/artifact/53acb957-63b5-46c2-8625-2f53f6574a69) | Fig. 7, earlier draft. |
| [Duck Trimmed Figure](https://claude.ai/code/artifact/a367d7b4-d82b-4e18-8d45-88adffa394b7) | Fig. 7 with the frame strip trimmed. |
| [Duck Camera Sheet](https://claude.ai/code/artifact/e8e3d4de-044d-4a60-9a3c-d9ddacb789ba) | Camera angle contact sheet for the duck. |
| [Duck Camera Options](https://claude.ai/code/artifact/edbb598b-6551-42b0-97fa-692c6cfb526d) | Earlier camera candidates for the same figure. |
| [Duck Figure Bench](https://claude.ai/code/artifact/7e7a4f55-6a61-465e-89de-d13cf8a07d7d) | Side-by-side bench of duck figure variants. |
| [Flicker Figure Draft](https://claude.ai/code/artifact/4234e4bd-c467-4973-a88b-0014f8a9ef90) | Fig. 6 flicker figure (chair, moss), frozen vs ours with insets. |

## Per-object result dailies — supplementary

One page per object/effect, our output rendered over the sequence. These are the
pool the gallery and supplementary result pages draw from.

| artifact | object · effect |
|---|---|
| [Bob — Spots](https://claude.ai/code/artifact/8b25ebcf-7fd8-4d03-828c-cb1d6e6c04eb) · [front view](https://claude.ai/code/artifact/95178fa8-506f-4450-8891-6399d7910351) · [slow](https://claude.ai/code/artifact/5ca03ec6-3ef5-40f3-8285-db6cad97a22a) | duck float, gliding spots (three camera/pacing variants) |
| [Teddy — Bleach](https://claude.ai/code/artifact/d36b17dc-8bea-4dcc-9f72-95550e7e486a) | teddy, bleach spreading |
| [Teddy — Fusion](https://claude.ai/code/artifact/e100fc81-ff27-4ed4-ad2b-23ae17294af8) | teddy, fusion effect |
| [TIE Fighter — Monochrome](https://claude.ai/code/artifact/c556ea7c-8588-45ee-8a5e-bfc0af15f2f1) | TIE fighter, monochrome |
| [Blub — Drying](https://claude.ai/code/artifact/769e5c0a-fb87-454f-b940-b978fae2d278) | blub, drying |
| [Blub — Rorschach](https://claude.ai/code/artifact/f53edf55-36f3-4bd4-9d51-c1a2cfb3de1b) | blub, Rorschach pattern |
| [Gargoyle — Spiral](https://claude.ai/code/artifact/cd9f8f14-da7f-4ea7-8b82-584a851b3e4f) | gargoyle, spiral |
| [Gargoyle — Effect One](https://claude.ai/code/artifact/56d50cb4-635d-46a6-9915-deb8d134e793) | gargoyle, effect one |
| [Goat — Flower](https://claude.ai/code/artifact/3ef792b0-4d9c-4f7f-993c-69a48a55c0ac) | goat, flowers |
| [Goat — Burnt](https://claude.ai/code/artifact/78fc9773-3529-472d-abe4-ed5d4d027082) | goat, burnt |
| [Goat — Clay](https://claude.ai/code/artifact/80391d05-6cba-40cc-ae2c-5b69c8003925) | goat, clay |
| [Octocat — Clay](https://claude.ai/code/artifact/182b4e85-6f2b-4b27-bb34-add537c5920e) | octocat, clay |
| [Octocat — Shine](https://claude.ai/code/artifact/a04d7456-2ff3-4eb1-9ffe-c4bdd8a1217e) | octocat, shine |
| [Moai — Animated](https://claude.ai/code/artifact/f933b248-d6d6-4248-9564-c7365ae756c7) | moai, animated |
| [Moai — Mosaic](https://claude.ai/code/artifact/1595a396-5385-4a9a-a615-27b08b171fe2) | moai, mosaic |
| [Moai — Silver](https://claude.ai/code/artifact/8f4e6759-da14-4351-b89e-2c1a5c5fadcc) | moai, silver |
| [Plane — Hokusai Waves](https://claude.ai/code/artifact/081e4d78-e28d-4d5e-ab25-5fab32582722) | airplane, Hokusai waves — the Fig. 1 effect |
| [Airplane — Red Cracks](https://claude.ai/code/artifact/4655989d-e245-4c25-a2b3-d784818368d3) | airplane, red cracks — in the Fig. 2 gallery |
| [Airplane — Blub](https://claude.ai/code/artifact/a5fa6f00-55b3-4e2a-b63e-77e009890752) | airplane, blub |
| [Fish — Glitter](https://claude.ai/code/artifact/1ab5ce84-0d5a-4a51-ae6d-ee66c428a851) | fish, glitter |
| [Dragon — Mushrooms](https://claude.ai/code/artifact/456e46ca-c9ef-4c20-9bc1-7bf550727f0e) | dragon, mushrooms |
| [Sheep — Mud](https://claude.ai/code/artifact/2c2a6846-812a-4f9d-85cb-115d33fa2f5d) | sheep, mud |
| [Sheep — Soil](https://claude.ai/code/artifact/17308bb4-e284-48cc-8cc7-f0803326df05) | sheep, soil |
| [Ivysaur — Petals](https://claude.ai/code/artifact/d8271192-1052-4d6b-a33d-fe55285af87b) | ivysaur, petals |
| [Ivysaur — Petals II](https://claude.ai/code/artifact/7a2212b7-9084-42d5-ab99-d1472803fe36) | ivysaur, petals, second effect |
| [Pegasus — Shine](https://claude.ai/code/artifact/47ff3926-62e5-4f62-b4d5-918738709a89) | pegasus, shine |
| [Pegaso — Effect One](https://claude.ai/code/artifact/9c305aa9-cb31-4572-bdf2-c67a886f2410) | pegaso, effect one |

## Rung-ladder comparisons — supplementary

Same object shown across the rung ladder, so the effect of each stage is visible
rather than only tabulated.

| artifact | object |
|---|---|
| [Rung Ladder Dailies](https://claude.ai/code/artifact/42db3ff0-0cc5-4ce7-92e1-0a03fec3b293) | index page across objects |
| [Octopus — Rust](https://claude.ai/code/artifact/6de0391a-422d-4a00-bcd8-1ec7e20deed9) | octopus, rust |
| [Octopus — Rainbow](https://claude.ai/code/artifact/b8272859-3623-425c-938f-3eb075b67082) | octopus, rainbow |
| [Blob — Orange Crack](https://claude.ai/code/artifact/9dcc41c7-d454-47bb-82c8-c887e42c54f7) | animal blob, orange crack |
| [Blob — Crack](https://claude.ai/code/artifact/31a51b72-7536-452d-a428-5ae9f79fd43a) | animal blob, crack |
| [Nefertiti — Effect 2](https://claude.ai/code/artifact/c5a2fcb1-cd0d-447e-b6ed-79810269b465) | ancient lady, effect 2 |

## Continuation experiments — supplementary

| artifact | one-liner |
|---|---|
| [Continuation Dailies](https://claude.ai/code/artifact/ce01e8b6-5457-48fe-be7c-8c2702e82191) | textured-continuation round 2 results (see the Textured continuation section above). |
| [Continuation Dailies (earlier)](https://claude.ai/code/artifact/9a9ed214-dcf0-4562-b52e-254c93d0b180) | first round of the same. |

**Caveat.** The artifact listing API returns at most 50 rows and is ordered by last
update, so pages not touched recently can fall outside it. This index is the union of
three listings taken on 2026-08-28/29 and is complete as of then; anything published
earlier and never re-published since may be missing. Re-check with `/artifacts`.

---

# Supplementary — status against Itai's list (2026-08-29)

> **SUPERSEDED.** See *Supplementary status, 2026-08-30 10:30* at the end of
> this file. Three counts in this section were wrong: r31 needed 40 render jobs
> rather than ~147 (only 24 of 42 objects were ever trained at rung31), the
> `view_*_31_*` figure of 21 was objects rather than renders, and the total was
> 376 jobs rather than 483.

Audited against disk, not memory. "Have" means the data exists and only assembly is
left; "need" means compute must run first.

## Figures

| item | status | what is missing |
|---|---|---|
| **Gallery, 4+ new meshes/effects** | ❌ need | New Kling reference videos + a 30-epoch fit per object. Nothing reusable on disk. |
| **Comparison, 2+ more** | ⚠️ partial | Renders exist for `27`, `27m`, `27e/f/g` (42 each). Baseline panels exist in `baselines4d/outputs`. Assembling more comparison figures needs no new compute. |
| **Window graph (flicker left axis, PSNR right, W on x)** | ✅ **have** | `out/window_ablation.json`, 42 objects, W=1/3/5/7/11. Pure plotting, zero compute. |
| **Window W=13 and W=15** | ❌ need | `mcfm_blend.py` `_OFFSETS` stops at `G` (W=11). Needs `H`/`I` added, then **84 training runs** (42 objects × 2 windows, ~3–6 h each). Biggest single item on the list. |
| **Existing textures, 1–2 more** | ❌ need | Same as gallery: new textured mesh + video + fit. |

## Tables

| item | status | what is missing |
|---|---|---|
| **Components off/on, texel-space** | ✅ **have** | `out/component_ladder_w11.json`, 24 objects. Artifact published. |
| **Components off/on, video-space** | ⚠️ **1 render batch away** | Needs the `r19` (cross-attn only) arm in pixel space: `view_*_19_*` = **0 on disk**. 168 render jobs (42 obj × 4 views). `27` and `27m` already rendered. |
| **Temporal attn + LoRA (= rung 37)** | ⚠️ **1 render batch away** | Texel done (42 objects). Pixel: `view_*_37_*` = **0**. 168 render jobs. |
| **Spatio-temporal attn (= rung 31)** | ⚠️ partial | Texel 24 objects; pixel `view_*_31_*` = **21 of 168**. ~147 render jobs to finish. |
| **LLM-as-a-judge** | ❌ need | No API key on the cluster (`ANTHROPIC_API_KEY` unset; `/usr/bin/ant` is Apache Ant, not the Anthropic CLI). Frame sampling is CPU-only and can run now; judging waits on a key from console.anthropic.com. Reference protocol: LL3M (ask Sining). |
| **More implementation detail** | ❌ need | Writing only. |

## Symbol discipline (Itai's note)

The temporal-attention variants must be described using the paper's own symbols:
`z_t` conditioning tokens, `z̃_t` the blend (Eq. 1), `α_δ^i` the weights (Eq. 2),
`W' = W + ΔW`, `ΔW = (α/r)BA` (Eq. 3), `S` structured latent, `D` decoder (Eq. 5).
State per variant which of those is duplicated, which is LoRA-adapted, and over how
many frames the attention runs.

## Immediate-launch order (no new training, no API key)

1. Window graph — no compute at all
2. `r19` pixel renders, 168 jobs — unlocks the video-space component table
3. `r37` pixel renders, 168 jobs — unlocks the temporal-attention table in video space
4. `r31` pixel renders, ~147 jobs — completes the spatio-temporal variant
5. LLM-judge contact sheets — CPU only, builds while everything else runs

Deferred because they need training, not just rendering: W=13/15 (84 runs), the new
gallery objects, and the extra textured-input examples.

## Supplementary run plan: video-space renders (COMPLETE 2026-08-30)

All three arms are in the queue. Submitter is
`experiments/dynamesh/jobs/submit_arm_renders.sh` (`ARM=r19|r37|r31|r31m`, plus
`DRY=1`, `ONLY=`, `LIMIT=`, `NICE=`), which drives the existing
`jobs/render_arm.sbatch`. No training: every job reloads a `lora_best.pt` already on
disk. One job per (object, view), four fixed cameras (train, diagA, diagB, diagC),
`turns=0` so the camera never moves and every frame-to-frame change is texture.

| arm | objects | jobs submitted | out tag | unlocks |
|---|---|---|---|---|
| `r19` cross-attn only | 42 of 42 | 168 | `view_<obj>_19_<view>` | video-space components off/on table |
| `r37` per-position temporal LoRA | 42 of 42 | 168 | `view_<obj>_37_<view>` | temporal attn + LoRA table in video space |
| `r31` spatio-temporal | 24 of 42 | 20 | `view_<obj>_31_<view>` | spatio-temporal row in video space |
| `r31m` same + MCFM v2_D | 24 of 42 | 20 | `view_<obj>_31m_<view>` | its MCFM counterpart |

**376 jobs, not 483.** Three corrections to the estimate in the section above, all
found by resolving run directories from `config.json` rather than trusting the
earlier count:

1. **r31 is 40 jobs, not ~147.** Only 24 objects have a 30-epoch rung31 checkpoint.
   The 18 batch-D/E objects (`moai_*`, `gargoyle_*`, `airplane_*`, `teddy_*`,
   `goat_*`, `ivysaur_*`, `blub_*`, `pegasus_shine`, `pegaso_effect_1`,
   `mosaic_painting`) were never trained at rung31, so the missing renders were
   never 42 objects' worth. 21 of the 24 were already rendered; the remaining 5
   objects x 4 views x 2 arms is the whole job. Rendering the other 18 would need
   training first, which puts them in the same bucket as W=13/15.
2. **`view_*_31_*` was 21 objects, not 21 renders.** 84 of 96 cells for `31` and 84
   of 96 for `31m` were already on disk.
3. **r37 could not be rendered by `render_arm.py` as it stood.** A rung37 checkpoint
   carries `tmix.*` (the per-position temporal mixer) on top of rung31's `blocks_t.*`
   and `gates.*`. `render_arm.py` mapped everything outside `{31, 33}` to
   `rung27_selfattn_lora`, whose registry has no `tmix`, so all 168 jobs would have
   died at `load_state_dict` before frame 1. Fixed by dispatching the RENDERER by
   rung as well as the trainer: rung 37 now runs `render_rung37_orbit.py`, which
   imports `rung37_perpos_temporal_lora` itself and needs no redirect.

### Outcome: all four arms complete, verified

376 of 376 jobs COMPLETED, zero failures, zero retries needed.

| arm | cells | verified |
|---|---|---|
| `r19` | 168/168 | every cell full-length; no two views of any object share a frame |
| `r37` | 168/168 | same check, same result |
| `r31` | 96/96 | complete |
| `r31m` | 96/96 | complete |

The second check is the one worth keeping: for each object the four cameras' first
frames were hashed and compared. Identical hashes across views would mean the cell
rendered four copies of one camera, which is what an empty `YAW0` produces and which
a frame COUNT cannot detect. Zero objects showed it on either arm.

So the video-space components table now has its missing `+ cross-attention` row in
pixels, resolved to the same checkpoint the published texel row used (checked object
by object against `out/TEXEL/<obj>_<arm>.json`: 42/42 agree for r19 and r37, 24/24 for
r31), and the temporal-attention table has its rung37 row.

**Duplicate fleet, for the record.** `jobs/submit_ladder_renders.sh` renders the same
cells into the same directories under `a19_`/`a31_`/`a37_` job names. It carried a
`C=$(ls "out/${TAG}/frames"/*.png | wc -l)` under `set -eo pipefail`, which exits 2 on
a missing directory and kills the script printing NOTHING -- the identical trap
`render_arm.sbatch` documents in its own header. Fixed, and it now also skips any cell
already queued under an `r19v_`/`r37v_`/`r31v_` name, because two renderers writing one
frames directory can tear a PNG mid-save.

### How it was launched

r19 and r37 had never been rendered in pixel space, so one canary of each went first
(`ancient_lady_effect_2`, train view) and the 334-job fan-out was held behind
`jobs/release_r19_r37_when_canary_ok.sh` until both were drawing frames. That gate is
the same discipline recorded in `render_arm.py`'s own header, where both of the bugs
that file exists to prevent were caught by a single canary. Both canaries passed; the
gate released. r31/r31m went straight out, because 84 cells had already been rendered
through that exact path.

Registered at submit time, not after the first silent failure: `r19v_`, `r37v_`,
`r31v_`, `r31mv_` are all four in `watchdog.py`'s `KIND` table (all four, because
`kind_of()` matches on `startswith` and `r31mv_` does not start with `r31v_`), every
job is in `out/job_manifest.json` and in `JOBLOG.tsv` with its pass condition, and
logs land in `out/RENDERS/<jobname>_<jobid>.log`. Re-running the submitter IS the
resubmit mechanism: it skips any cell whose frame count is already complete and any
whose job name is still queued.

### Still not launched

* **W=13 and W=15**: 84 training runs, and `_OFFSETS` in `mcfm_blend.py` needs `H`/`I`
  first. The fidelity curve is already flat from W=5 to W=11 with differences that are
  not significant, so this most likely extends a flat line; the argument for running it
  is closing the question rather than leaving a reviewer to ask where the curve turns.
* **Gallery objects and extra textured inputs**: need new Kling videos and fresh fits.
* **LLM-as-a-judge**: needs an API key on the cluster. Frame sampling is CPU-only and
  can run before the key arrives.
* **Window graph**: needs no cluster at all. `out/window_ablation.json` already holds
  42 objects at W=1/3/5/7/11, so it is a plotting task of about ten minutes.

## Submitted 2026-08-29 — pixel-space ladder renders

338 render jobs, one per (object, arm, view), job names `a<arm>_<obj>_<view>`,
script `jobs/render_arm.sbatch`, submitter `jobs/submit_ladder_renders.sh`.
All logged to `JOBLOG.tsv` and `out/job_manifest.json`; submission record in
`out/logs/sub_ladder.log`. Renders land in `out/view_<obj>_<arm>_<view>/frames`.

| arm | rung | what it is | submitted | coverage after |
|---|---|---|---|---|
| `19` | rung 19 | cross-attention LoRA only | 167 | 42/42 objects |
| `37` | rung 37 | + learned per-position temporal LoRA | 167 | 42/42 objects |
| `31` | rung 31 | + joint spatio-temporal attention | 4 | **24/42 objects** |

**Run resolution.** Each arm's checkpoint is read from the same
`out/TEXEL/<obj>_<arm>.json` the published texel ladder used, not from a glob over
`runs/`. This matters: rung 19 has **29 objects with more than one 30-epoch run**
(some `mcfm=None`, some `mcfm=v2_D`), so a glob would have silently picked a
different checkpoint than the texel row and the two tables would not correspond.
Verified before submitting: all 42 r19 runs are rung19 / 30 epochs / `mcfm=None`
with a checkpoint, and all 42 r37 runs are rung37 / 30 epochs.

**rung 31 is short by 18 objects and rendering cannot fix it.** These were never
trained, so there is no checkpoint to render:

> airplane_blub, airplane_red_cracks, blub_drying, blub_raurshaw,
> gargoyle_effect_one, gargoyle_spiral, goat_burnt, goat_clay, goat_flower,
> ivysaur_petal, ivysaur_petal_2, moai_animated, moai_silver, mosaic_painting,
> pegaso_effect_1, pegasus_shine, teddy_bleach, teddy_fusion

Options: report the spatio-temporal row on its own 24-object common set and say so,
or train those 18 first (18 runs × ~3–6 h) to put every ladder row on the same 42.
The first is honest and free; the second is cleaner and costs a day of GPU.

**Next step once these land:** run the full-rate metrics per arm
(`jobs/fullrate_metrics_arm.py --arm 19|31|37`) into `out/FULLRATE_A19` /
`_A31` / `_A37`, then aggregate with `jobs/fullrate_table_scoped.py`. That produces
the video-space component table where the all-components row matches Table 1.

---

# Supplementary status, 2026-08-30 10:30

Supersedes the 2026-08-29 status section above. Everything here was read off disk
and off `sacct`, not from memory.

## What is finished

| item | state | evidence |
|---|---|---|
| Video-space renders, arms r19 / r37 / r31 / r31m | **done** | 376/376 jobs COMPLETED |
| Video-space metrics for those arms | **done** | 129/129 jobs, `out/FULLRATE_R19` / `_R37` / `_R31` / `_R31M` |
| W=15 (`v2_I`) training | **done** | 42/42 cells with `final_eval.json` |
| W=13 (`v2_H`) training | **41/42** | last cell `ivysaur_petal_2` resuming from ckpts |
| pumpkin_rot renders | **restored** | 4/4 views on all four arms |

Zero genuine failures across all three fleets. The only `FAILED` rows in the window
are the 12 pumpkin_rot duplicates described below, which are superseded.

## Video-space component and temporal tables (the numbers)

Averaged over the four fixed cameras, all 150 frames, PSNR/SSIM at the training view
against the 2D copy. Same driver, masks, panel crops and reference as the window arms,
so these rows and the window rows belong in one table.

| row | n | flicker | accel | drift | PSNR | SSIM |
|---|---|---|---|---|---|---|
| frozen | 41 | 0.01807 | 0.02708 | 0.3356 | 12.00 | 0.296 |
| + cross-attention (r19) | 41 | 0.00767 | 0.01084 | 0.2175 | 23.66 | 0.752 |
| + self-attention (r27 = W3) | 42 | 0.00613 | 0.00678 | 0.2162 | 24.90 | 0.798 |
| + temporal (MCFM W=11) | 42 | 0.00545 | 0.00570 | 0.2172 | **24.89** | **0.798** |
| temporal attn + LoRA (r37) | 41 | 0.00727 | 0.01000 | 0.2112 | 24.81 | 0.798 |
| spatio-temporal attn (r31) | 23 | 0.00887 | 0.01232 | 0.2371 | 24.21 | 0.809 |
| r31 + MCFM | 24 | 0.00732 | 0.00794 | 0.2421 | 24.49 | 0.812 |

**The all-components row matches the paper**, which was Itai's stated requirement for
this table: 24.89 / 0.798 in video space against the published Table 1 reference of
24.90 / 0.798. Agreement to two decimals also confirms the 2D-copy reference was used
rather than the raw-frame one, which reads about 6 dB low.

**The ladder reproduces the texel ordering in pixels.** Flicker falls monotonically at
every step, 0.01807 to 0.00545, a 70% total reduction, with fidelity saturating after
cross-attention.

**Parameter-free MCFM beats both learned temporal variants in video space**, at equal
or better fidelity: flicker 0.00545 for W=11 against 0.00727 for temporal attention
plus LoRA and 0.00887 for spatio-temporal attention. This is the same conclusion the
texel measurements reached, now on the support where baselines can also be scored.

**Caveat that must appear in the caption:** the r31 rows average over 23 to 24 objects,
not 42, because the batch D/E objects were never trained at rung31. That row is not
comparable to the 42-object rows unless the object set is stated.

## W=13 / W=15: what it took

`_OFFSETS` in `mcfm_blend.py` stopped at `G` (W=11), so these windows were not merely
unrun, they were unrunnable: `parse_mode` would have raised `KeyError` on `H`. Added
as symmetric tuples, `H` spanning -6..+6 and `I` spanning -7..+7, plus `MODES` and the
`temporal_only_w13` / `_w15` aliases. Trainers read their `--mcfm` choices from that
module rather than a copy, so the change propagates without touching them.

Verified on CPU before a single job was submitted: both widths parse with the centre
at index 0, and `blend_conds` preserves shape and finiteness on a 150-frame sequence,
a 121-frame one, and an 8-frame one, the last being the real edge case where the
window is wider than the sequence and every neighbour clamps.

Two canaries then ran first and the other 82 were held behind a gate keyed on the
trainer's own `[MCFM] GATE-blend max|blended-vanilla| ... (must be > 0)` line plus
`[GATE-grad] PASSED`. Those land within minutes rather than after 3 to 6 hours, and
they catch the failure that actually matters here: an offsets tuple that parses but
does not widen would have trained 84 jobs that were all secretly W=11. The gate
confirmed the real widths from the logs.

Training took about 8 hours for all 84, not the 10 to 14 estimated. The estimate
assumed a timed-out cell needed a fresh 4 hour allocation; in practice a resume from
`ckpts` finished the remaining epochs in 15 to 20 minutes. 31 of 84 cells timed out at
the 4 hour wall at least once, which is the documented normal path, not a failure.

Early signal, to be held lightly until all 84 are aggregated: on spot_lava, W=13 gives
24.330 dB and W=15 gives 24.343 dB, within 0.013 dB of each other, which is what the
already-flat W=5 to W=11 curve predicted.

## The pumpkin_rot incident, and the bug behind it

`jobs/submit_ladder_renders.sh` read the frame count with
`awk -F"\t" -v o="$OBJ" '$1==o{print $3}'`. Column 1 of `jobs/rung37_objects.tsv` is
the BATCH LETTER and column 2 is the object, so `$1==o` never matched and NFR fell
back to 150 for every object. Harmless for the 41 objects that really are 150 frames,
fatal for pumpkin_rot at 121: the renderer walked off the end of `frames_from_video`
at `frame_0122`, died, and took the completed frames with it, destroying 12 already
finished cells at arms 19, 37 and 31. Fixed to `$2==o`, the cells were re-queued with
`NFR=121`, and all four views are back on all four arms.

The same script also carried `C=$(ls "out/${TAG}/frames"/*.png | wc -l)` under
`set -eo pipefail`. A missing directory makes `ls` exit 2, pipefail propagates it, and
`set -e` kills the script **printing nothing**. This is the identical trap
`render_arm.sbatch` documents in its own header. Fixed, and the script now also skips
any cell already queued under an `r19v_` / `r37v_` / `r31v_` name, because two
renderers writing one frames directory can tear a PNG mid-save.

## Still outstanding

| item | blocker |
|---|---|
| **Texel metrics for W=13/15** | **84 GPU jobs, not yet submitted.** Training writes PSNR/SSIM only; flicker, accel and drift come from a separate ODE decode over the PBR voxel field. `out/TEXEL/*_w13.json` and `*_w15.json` are empty, so the window graph cannot gain its last two points until this runs. |
| Full-rate reference for W=13/15 | CPU-only, `fw13_` / `fw15_`, so the new points share the 2D-copy reference with W=3/5/7/11 |
| Window graph regeneration | needs the two rows above; then it is a ten-minute plot |
| Video-space table builder | `jobs/build_window_table.py` is the pattern; swap its `WINDOWS` list for the `FULLRATE_R*` dirs |
| Gallery, 4+ new meshes | new Kling videos plus 30-epoch fits; nothing reusable on disk |
| Existing textures, 1 to 2 more | same blocker as the gallery |
| LLM as a judge | no `ANTHROPIC_API_KEY` on the cluster; contact-sheet sampling is CPU-only and could build now |
| Variant text and paper symbols | writing only, nothing blocking |
| More implementation detail | writing only |

## Scripts added this session

| file | what it does |
|---|---|
| `jobs/submit_arm_renders.sh` | video-space renders for r19/r37/r31/r31m, one job per (object, view) |
| `jobs/submit_ladder_fullrate.sh` | video-space metrics for those arms, CPU-only |
| `jobs/submit_window_hi.sh` | W=13/15 training across all 42 objects |
| `jobs/release_r19_r37_when_canary_ok.sh` | canary gate for the render fan-out |
| `jobs/release_window_hi_when_canary_ok.sh` | canary gate for the W=13/15 fan-out |
| `jobs/watch_arm_renders.py` | render-fleet failure monitor with idempotent self-heal |
| `jobs/watch_window_hi.py` | training-fleet monitor; TIMEOUT counted, never announced |

`render_arm.py` also gained renderer dispatch by rung: a rung37 checkpoint carries
`tmix.*` on top of rung31's `blocks_t.*` and `gates.*`, and everything outside
`{31, 33}` was mapped to `rung27_selfattn_lora`, whose registry has no `tmix`. All 168
r37 jobs would have died at `load_state_dict` before frame 1. rung 37 now dispatches
to `render_rung37_orbit.py`, which imports `rung37_perpos_temporal_lora` itself.

All new job-name prefixes are registered in `watchdog.py`'s `KIND` table at submit
time: `r19v_`, `r37v_`, `r31v_`, `r31mv_`, `fr19_`, `fr37_`, `fr31_`, `fr31m_`. Each
is spelled out rather than shortened, because `kind_of()` matches on `startswith` and
`fr31m_` does not start with `fr31_`, the same way `rarm4c_` slipped past `rarm_`.

## Temporal-attention ablation: which rung is which, and the 11-frame gap

Itai's two variants map to two rungs. Both descriptions below are what the code
actually does, read from `config.json`, not from the run-directory name.

| Itai's wording | rung | what it does | window on disk | trained |
|---|---|---|---|---|
| "temporal attention with LoRA, weights duplicated from the TRELLIS cross-attention weights" | **37** | a second cross-attention branch initialised as a copy of the frozen TRELLIS cross-attention, LoRA-adapted, attending token j to token j across the window (`temporal_geometry: per_position`) | **W=3** | 42/42 |
| "spatio-temporal attention, TRELLIS cross-attention weights with LoRA applied to the tokens of 11 frames" | **31** | one dual-branch registry pooling all W x N tokens into a single softmax over time and space, LoRA on the cross-attention weights | **W=3** | 24/42 |

`rung33` is rung31 plus KL, also W=3, 24/42. `rung31+MCFM` is rung31 with the
parameter-free `v2_D` blend on top, also W=3, 24/42.

### The discrepancy

Itai specifies **11 frames** for the spatio-temporal variant. **Every temporal-attention
run on disk is a 3-frame window** -- rung31, rung31+MCFM, rung33 and rung37 alike.
Nothing at `temporal_window >= 5` exists anywhere in `runs/`. The 11 in the paper
belongs to **MCFM's** window (`v2_G`), which is the parameter-free operator, not the
learned attention. So the published W=11 number and the learned spatio-temporal
variant were never the same window, and the text must not imply they were.

### W=11 was unrunnable, not merely unrun

`rung31_dual_attn_lora.py:367` declared `--temporal-window ... choices=[0, 3, 5]`, so
`--temporal-window 11` exited at argument parsing. Same shape of problem as MCFM's
`_OFFSETS` stopping at `G`. The stacking underneath was always width-generic
(`_half = W // 2`, offsets `-half..+half`, end-clamping identical to
`mcfm_blend.blend_conds`), so the fix was the choices list plus making
`jobs/fig31.sbatch` take `${TW:-3}`, which leaves every existing W=3 path untouched.

Verified on CPU before any job was submitted:

```
W= 3: offsets [-1,0,1]     tokens 1029 ->  3087  centre=frame f OK  ends clamp OK
W=11: offsets [-5,...,+5]  tokens 1029 -> 11319  centre=frame f OK  ends clamp OK
```

The centre-slice check is the one that matters: the trainer's own `GATE-window`
asserts the middle slice is frame f byte-for-byte, and a wrong offset list would break
the spatial branch silently rather than loudly.

### Status: submitted 2026-08-30 10:45, canary-gated on MEMORY

`jobs/submit_rung31_w11.sh`, 42 cells, job names `31_w11_<obj>` (which start with
`31_`, so `watchdog.py` already watches them and resubmits through `fig31.sbatch`
with `TW=11` preserved in the manifest export). Run dirs land as
`rung31_l1_lp_tw11b_*`, distinct from `tw3b`, so nothing can resume across windows.

Canary `31_w11_spot_lava` first, the other 41 held behind
`jobs/release_r31w11_when_canary_ok.sh`.

**This gate checks VRAM, which the W=13/15 gate did not need to.** MCFM's blend is
precomputed once over the cached conditioning, so a wider window costs setup and
nothing per step. rung31 is the opposite: the context tensor itself grows 3.67x, and
W=3 already peaks at **24 GiB of a 48 GiB card**. The gate refuses to release unless
it sees the `[R31] window=11 ... tokens 1029 -> 11319` line, a passing GATE-window,
and a measured peak under 42 GiB across at least 20 readings. A blind fan-out could
have been 42 simultaneous CUDA OOMs.

If the gate fails on memory, the fallback is W=7 or W=9; the trainer now accepts
`[0, 3, 5, 7, 9, 11]`.

### Open decision, for Itai

Two readings of the 11, and they cost very differently:

1. He means the learned spatio-temporal variant really should be 11 frames. That is
   the 42 runs now submitted, plus the question of whether rung37 should match.
2. He is recalling MCFM's W=11 and the learned variants were always meant to be W=3.
   Then the fix is one sentence stating W=3 explicitly, and the 42 runs are a bonus
   data point rather than a requirement.

**Also still open:** the 18 batch-D/E objects have no rung31 run at ANY window, so the
W=3 spatio-temporal row averages over 24 objects while every other row averages over
42. Either say so in the caption, or train those 18.

---

# Meshes suggested by Itai (2026-08-30)

Itai wants **a car result in the submission**: "Try also rust on a car. Or other cool
effect on a car." He pointed at `nascar.obj` in
<https://github.com/eladrich/latent-nerf/tree/main/shapes>, and separately at Dale's
robot: "Rust on this robot would be cool (look for it under Dale's folder)."

## Both meshes are already on the cluster

The whole `eladrich/latent-nerf/shapes` folder is mirrored locally, so nothing needs
downloading. `nascar.obj` there is **byte-identical** to the GitHub file (md5 `28f1c7c1`).

| mesh | local path | verts | tris | boundary edges | watertight |
|---|---|---|---|---|---|
| **nascar** | `ddecatur/latent-nerf/shapes/nascar.obj` | 3,750 | 7,500 | **0** | **YES** |
| **robot** | `ddecatur/3DHighlighter/data/shapes/robot.obj` | 3,136 | 6,132 | **0** | **YES** |

The same `shapes/` folder also carries blub, bob, bunny, camel, cabin, desk_chair, dog,
giraffe, hand, lego_minifig, penguin, person, potion and spot, several of which are
already in our 42. Copies of the robot exist in four places under `ddecatur/`; three are
byte-identical (md5 `ea5f96bd`) and all four are watertight.

Local copy for convenience: `supplementary_meshes/OBJ/robot_dale.obj`.

## Why the car we already had could not be used

`supplementary_meshes/OBJ/car.obj` is the **same file** as
`itailang/mesh_feature_fields/meshes/car.obj` — identical counts — and it carries **409
open boundary edges**. Those are the white slivers on the hood, roof and flank in
`front/car.png`. Kling paints through holes, so it would have produced pattern showing
through the body. Every other vehicle mesh in the lab is worse:

| mesh | tris | boundary edges |
|---|---|---|
| `ddecatur/latent-nerf/shapes/nascar.obj` | 7,500 | **0** |
| `itailang/mesh_feature_fields/meshes/car.obj` (= ours) | 31,812 | 409 |
| `itailang/.../truck.obj` | 32,725 | 412 |
| `itailang/geocap/meshes/cars/car.obj` | 1,500 | 796 |
| `ddecatur/analysis-via-synthesis/.../car_2.obj` | 98,752 | 72,432 |

Searched: the ranalab filesystem (itailang, ddecatur, danielfu, guanc, sininglu, brian)
and the full 800K-object Objaverse index via its metadata cache at
`/net/projects2/ranalab/objaverse`. The nascar is the only clean vehicle.

## Rendered inputs and prompts, ready to shoot

| object | input image | camera | prompt |
|---|---|---|---|
| nascar | `out/MESH_CANDIDATES/nascar.png` | yaw 60, pitch 36, 84% exposed | see note below |
| robot | `out/MESH_CANDIDATES/robot_dale.png` | yaw 330, pitch 8, 70% exposed | `supplementary_meshes/KLING_PROMPTS_ROBOT.md` |
| car (holey, superseded) | `supplementary_meshes/front/car.png` | up −Z, yaw 70, pitch 30 | `supplementary_meshes/KLING_PROMPTS_CAR.md` |

The car prompt was written against the holey mesh and needs re-pointing at nascar before
use; its ordering block (seams, sill, arch lips, stone chips) still applies, but nascar
is a smoother body with fewer panel joins.

## Which effect for the car

Rust is the obvious answer and it is a good one, but note the tradeoff. Rust needs
**nucleation sites** to look like a process rather than an optimisation, and nascar's
smooth low-poly body offers fewer than a panelled car would: the wheel arches and the
lower sill are the strongest it has. The robot, by contrast, offers feet and ankles,
every joint, panel edges and fastener recesses, which is why the rust prompt written for
it can state a much stronger causal order — corrosion climbing **up** from the feet and
**out** from the joints, head and upper chest last.

So the plan is: **rust on the robot** as the strong result, and for the car either rust
concentrated on arches and sill, or an effect that does not depend on seams. Candidates
for the latter, none of them used yet in any round:

* **anodised heat tint** — straw to bronze to purple to blue sweeping across the panels
  as a temperature front. Directional, irreversible, and the colour order is the clock.
* **dust and desert patina** settling on upward-facing surfaces first, which makes
  gravity the ordering principle instead of seams.

Both fit a smooth body better than rust does.

## Status

Meshes located, audited for watertightness, rendered at solved best views, prompts
written. **Blocked on Kling videos** — one per object, then a 30-epoch fit each.
