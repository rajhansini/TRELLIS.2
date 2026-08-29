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
