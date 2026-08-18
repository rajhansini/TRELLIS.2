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
