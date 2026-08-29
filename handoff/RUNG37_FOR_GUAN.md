# rung 37 — per-position temporal LoRA — handoff for Guan

**Date** 2026-08-26 · **Owner** Raj · **Status** implemented, smoke-passed, not yet run at scale

---

## What rung 37 is

Three LoRA sets on one backbone:

1. **cross-attention LoRA** — `to_q`, `to_kv`, `to_out`
2. **self-attention LoRA** — `sa_qkv`, `sa_out`
3. **temporal LoRA** — a *second* pass through the **same frozen cross-attention weights**,
   carrying its own separate adapter and a learned per-block gate

`out = S + gate * T`, gate initialised to 0 so step 0 is bit-identical to rung 27.

**The temporal pass attends to one spatial position across the frames of the window.**
Token *i* of frame *t* sees token *i* of frames *t-1, t, t+1* and nothing else. There is no
spatial mixing in that branch — spatial mixing is left to the model's own attention.

### Why it is not rung 31

rung 31's temporal pass hands the flow the whole stacked window (`W x 1029` tokens) with no
mask, so every voxel attends over every spatial position in every frame — joint
spatio-temporal, i.e. MCFM `v3` geometry. rung 37 folds the window to `(N, W, D)` and
collapses the `W` axis **per position** first, so the branch has MCFM `v2_D` geometry.

rung 37 **initialises exactly at MCFM v2_D** (verified: max abs difference 0.0). Q and K are
LoRA deltas on the identity with B zero-init, so at step 0 the learned operator *is* the
parameter-free blend, and training can only move it away. That makes MCFM the initialisation
of the learned operator rather than a separate baseline — the cleanest possible control.

### Collision safety

rung 37 stamps `rung: 37` and `temporal_geometry: 'per_position'` into the hashed config, and
its label is `rung37_l1_lp_tw3bpp_...`. It cannot share a directory with, or resume a
checkpoint from, rung 31. **Do not remove `temporal_geometry` from the config** — without it
the two rungs hash identically.

---

## Files

| file | what |
|---|---|
| `experiments/dynamesh/rung37_perpos_temporal_lora.py` | the trainer |
| `experiments/dynamesh/jobs/rung37_perpos.sbatch` | one object per job |
| `experiments/dynamesh/jobs/submit_rung37.sh` | fleet submitter, idempotent |
| `experiments/dynamesh/jobs/rung37_objects.tsv` | **the object/path manifest below** |

## Running it

```bash
cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
DRY=1 jobs/submit_rung37.sh              # print the work list, submit nothing
jobs/submit_rung37.sh                    # all 42 objects
BATCH=D,E jobs/submit_rung37.sh          # only batches D and E
ONLY=spot_lava,chair_moss jobs/submit_rung37.sh
```

Idempotent: an object with a finished `final_eval.json`, or a job already queued, is skipped.
Re-run after failures and it submits only what is still missing.

Single object by hand:

```bash
sbatch --job-name=r37_chair_moss \
  --export=ALL,OBJ=chair_moss,NFR=150,\
MESH=/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_moss/mesh/chair_moss_render_frame.obj,\
GTDIR=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_chair_moss/frames \
  jobs/rung37_perpos.sbatch
```

## Constraints that matter

- **`--constraint="a40|L40S"`** — the trellis2 env dies on a30 with `GLIBC_2.32 not found`
  importing `o_voxel._C`. It lives in the sbatch, not the command line, because a requeue
  drops CLI flags.
- **`--mem=32G`**, measured peak RSS 15.3G. 96G throttles the fleet under the group TRES cap.
- **`NFR` comes from the manifest, never hardcode 150.** `pumpkin_rot` is **121** frames and
  is on the `_guan` mesh; hardcoding 150 trains it against the wrong targets, completes
  successfully, and is silently wrong.
- Walltime 4h. rung 31 took ~2-4h per object at 150 frames; rung 37 is comparable.

## Measuring afterwards

Texel metrics come from `jobs/texel_after_train.sbatch` with `RUNG=37 MODE=None
TAG=<obj>_r37`. It runs `--skip-render`, so there are no PNGs and no ffmpeg — the ODE decode
writes one JSON. PSNR/SSIM come from that run's `final_eval.json` (`['final']` for the
adapted arm, `['frozen']` for the frozen one).

## Verification already done

- initialises to MCFM v2 exactly (maxdiff 0.0); per-position isolation confirmed
  (corrupting position 7 changes only position 7)
- frame ordering: `unflatten` recovers `[t-1, t, t+1]` as `_stack` wrote them; `win[mid]`
  equals the spatial slice bit for bit
- 491,520 mixer parameters land in `reg.parameters()`; gates and all B matrices zero at init
- **smoke run passed on chair_moss** (job 2214117, 2 epochs / 8 frames): `GATE-window PASSED`,
  `DualCrossAttn installed on 30 blocks`, `GATE 0 param count + B=0 identity PASSED`,
  `GATE-grad PASSED — every B matrix receives gradient`,
  `GATE-plain max |frozen - lora(B=0)| = 0.000e+00`

---

## Object manifest — 42 objects, batches A-E

All paths verified present on 2026-08-26; mesh, conditioning frames and 2D-copy targets all
exist with frame counts matching `n_frames`.

- **gt_dir** = `data/<obj>/frames_from_video` — the driving clip, the conditioning
- **gt_render_dir** = `out/gt_targets_<obj>/frames` — the 2D-copy targets used as the loss target

| batch | count |
|---|---|
| A | 4 |
| B | 11 |
| C | 9 |
| D | 11 |
| E | 7 |
| **total** | **42** |

| batch | object | frames | mesh | gt_render_dir |
|---|---|---|---|---|
| A | `ancient_lady_effect_2` | 150 | `data/ancient_lady_effect_2/mesh/ancient_lady_effect_2_render_frame.obj` | `out/gt_targets_ancient_lady_effect_2/frames` |
| B | `animal_blob_crack` | 150 | `data/animal_blob_crack/mesh/animal_blob_crack_render_frame.obj` | `out/gt_targets_animal_blob_crack/frames` |
| B | `animal_blob_orange_crack` | 150 | `data/animal_blob_orange_crack/mesh/animal_blob_orange_crack_render_frame.obj` | `out/gt_targets_animal_blob_orange_crack/frames` |
| B | `chair_ice` | 150 | `data/chair_ice/mesh/chair_ice_render_frame.obj` | `out/gt_targets_chair_ice/frames` |
| B | `chair_moss` | 150 | `data/chair_moss/mesh/chair_moss_render_frame.obj` | `out/gt_targets_chair_moss/frames` |
| B | `chair_real_wooden_crack` | 150 | `data/chair_real_wooden_crack/mesh/chair_real_wooden_crack_render_frame.obj` | `out/gt_targets_chair_real_wooden_crack/frames` |
| C | `dragon_mush` | 150 | `data/dragon_mush/mesh/dragon_mush_render_frame.obj` | `out/gt_targets_dragon_mush/frames` |
| C | `dragon_mush2` | 150 | `data/dragon_mush2/mesh/dragon_mush2_render_frame.obj` | `out/gt_targets_dragon_mush2/frames` |
| C | `fish_glitter` | 150 | `data/fish_glitter/mesh/fish_glitter_render_frame.obj` | `out/gt_targets_fish_glitter/frames` |
| C | `fish_ink` | 150 | `data/fish_ink/mesh/fish_ink_render_frame.obj` | `out/gt_targets_fish_ink/frames` |
| A | `hand_rorschach` | 150 | `data/hand_rorschach/mesh/hand_rorschach_render_frame.obj` | `out/gt_targets_hand_rorschach/frames` |
| B | `napolean_teapot_crack` | 150 | `data/napolean_teapot_crack/mesh/napolean_teapot_crack_render_frame.obj` | `out/gt_targets_napolean_teapot_crack/frames` |
| B | `napolean_waves` | 150 | `data/napolean_waves/mesh/napolean_waves_render_frame.obj` | `out/gt_targets_napolean_waves/frames` |
| C | `octocat_clay` | 150 | `data/octocat_clay/mesh/octocat_clay_render_frame.obj` | `out/gt_targets_octocat_clay/frames` |
| C | `octocat_shine` | 150 | `data/octocat_shine/mesh/octocat_shine_render_frame.obj` | `out/gt_targets_octocat_shine/frames` |
| B | `octopus_rainbow` | 150 | `data/octopus_rainbow/mesh/octopus_rainbow_render_frame.obj` | `out/gt_targets_octopus_rainbow/frames` |
| B | `octopus_rust` | 150 | `data/octopus_rust/mesh/octopus_rust_render_frame.obj` | `out/gt_targets_octopus_rust/frames` |
| B | `octopus_sparkle` | 150 | `data/octopus_sparkle/mesh/octopus_sparkle_render_frame.obj` | `out/gt_targets_octopus_sparkle/frames` |
| B | `octopus_tar` | 150 | `data/octopus_tar/mesh/octopus_tar_render_frame.obj` | `out/gt_targets_octopus_tar/frames` |
| A | `pumpkin_rot` | 121 | `data/pumpkin_rot/mesh/pumpkin_rot_render_frame_guan.obj` | `out/gt_targets_pumpkin_rot_guan/frames` |
| C | `sheep_mud` | 150 | `data/sheep_mud/mesh/sheep_mud_render_frame.obj` | `out/gt_targets_sheep_mud/frames` |
| C | `sheep_soil` | 150 | `data/sheep_soil/mesh/sheep_soil_render_frame.obj` | `out/gt_targets_sheep_soil/frames` |
| A | `spot_lava` | 150 | `data/spot_lava/mesh/spot_render_frame.obj` | `out/gt_targets_spot_lava/frames` |
| C | `tie_fighter_bw` | 150 | `data/tie_fighter_bw/mesh/tie_fighter_bw_render_frame.obj` | `out/gt_targets_tie_fighter_bw/frames` |
| D | `moai_silver` | 150 | `data/moai_silver/mesh/moai_silver_render_frame.obj` | `out/gt_targets_moai_silver/frames` |
| D | `moai_animated` | 150 | `data/moai_animated/mesh/moai_animated_render_frame.obj` | `out/gt_targets_moai_animated/frames` |
| D | `gargoyle_spiral` | 150 | `data/gargoyle_spiral/mesh/gargoyle_spiral_render_frame.obj` | `out/gt_targets_gargoyle_spiral/frames` |
| D | `gargoyle_effect_one` | 150 | `data/gargoyle_effect_one/mesh/gargoyle_effect_one_render_frame.obj` | `out/gt_targets_gargoyle_effect_one/frames` |
| D | `airplane_blub` | 150 | `data/airplane_blub/mesh/airplane_blub_render_frame.obj` | `out/gt_targets_airplane_blub/frames` |
| D | `airplane_red_cracks` | 150 | `data/airplane_red_cracks/mesh/airplane_red_cracks_render_frame.obj` | `out/gt_targets_airplane_red_cracks/frames` |
| D | `teddy_bleach` | 150 | `data/teddy_bleach/mesh/teddy_bleach_render_frame.obj` | `out/gt_targets_teddy_bleach/frames` |
| D | `teddy_fusion` | 150 | `data/teddy_fusion/mesh/teddy_fusion_render_frame.obj` | `out/gt_targets_teddy_fusion/frames` |
| D | `goat_burnt` | 150 | `data/goat_burnt/mesh/goat_burnt_render_frame.obj` | `out/gt_targets_goat_burnt/frames` |
| D | `goat_flower` | 150 | `data/goat_flower/mesh/goat_flower_render_frame.obj` | `out/gt_targets_goat_flower/frames` |
| D | `goat_clay` | 150 | `data/goat_clay/mesh/goat_clay_render_frame.obj` | `out/gt_targets_goat_clay/frames` |
| E | `ivysaur_petal` | 150 | `data/ivysaur_petal/mesh/ivysaur_petal_render_frame.obj` | `out/gt_targets_ivysaur_petal/frames` |
| E | `ivysaur_petal_2` | 150 | `data/ivysaur_petal_2/mesh/ivysaur_petal_2_render_frame.obj` | `out/gt_targets_ivysaur_petal_2/frames` |
| E | `blub_raurshaw` | 150 | `data/blub_raurshaw/mesh/blub_raurshaw_render_frame.obj` | `out/gt_targets_blub_raurshaw/frames` |
| E | `blub_drying` | 150 | `data/blub_drying/mesh/blub_drying_render_frame.obj` | `out/gt_targets_blub_drying/frames` |
| E | `pegasus_shine` | 150 | `data/pegasus_shine/mesh/pegasus_shine_render_frame.obj` | `out/gt_targets_pegasus_shine/frames` |
| E | `pegaso_effect_1` | 150 | `data/pegaso_effect_1/mesh/pegaso_effect_1_render_frame.obj` | `out/gt_targets_pegaso_effect_1/frames` |
| E | `mosaic_painting` | 150 | `data/mosaic_painting/mesh/mosaic_painting_render_frame.obj` | `out/gt_targets_mosaic_painting/frames` |

Machine-readable copy: `experiments/dynamesh/jobs/rung37_objects.tsv` (tab separated, absolute paths).
