# generalization_jobs.md

Zero-shot effect transfer. A LoRA fitted on ONE object is loaded read-only and rendered
on a DIFFERENT mesh. No optimizer, no backward, no gradient at transfer time --
GATE-notrain asserts it process-wide, GATE-transfer refuses a mesh the adapter was
trained on.

THE CONDITIONING IS THE SOURCE OBJECT'S OWN VIDEO, never the target's. The renderer
pulls frame i at every step, so the motion comes from the conditioning advancing, not
from the adapter. Feed a still and the output is static. The claim under test is "the
adapter lifts a 2D dynamic texture onto an arbitrary 3D shape", NOT "the adapter invents
a texture from nothing". Anyone reading these figures as the stronger claim is wrong.

Renderer: experiments/dynamesh/render_rung31_transfer_mcfm.py  (--mcfm optional; with it
unset the code path is identical to render_rung30_transfer.py)
Launcher: experiments/dynamesh/jobs/r32_transfer_any.sbatch
Every job: --sweep both --n-frames 150 --turns 0, four fixed yaws 000/090/180/270.
yaw000 is the training view; 090/180/270 were never supervised.
Output panels: TRELLIS.2 frozen | adapted.

## Source LoRAs (all pre-existing; none trained for these experiments)

| effect | arm | run | conditioning frames |
|---|---|---|---|
| lava | vanilla | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94` | `data/spot_lava/frames_from_video` |
| lava | MCFM v2_D | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_59c51f6e` | `data/spot_lava/frames_from_video` |
| rorschach (spot) | vanilla | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_9e09dbed` | `data/spot_raurshaw/frames_from_video` |
| rorschach (spot) | MCFM v2_D | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_72244c6e` | `data/spot_raurshaw/frames_from_video` |
| rorschach (hand) | vanilla | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_1eef6f88` | `data/hand_rorschach/frames_from_video` |
| rorschach (hand) | MCFM v2_D | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_eaa1fb67` | `data/hand_rorschach/frames_from_video` |

Configs are byte-identical across all six on 23 fields (rung 27, qkvo+sa, blocks all,
rank 4, alpha 4.0, L1+LPIPS w=0.1, 30 epochs, lr 1e-4, seed 42, 150 frames, res 512,
render 960, loss_region rendered_gt). The ONLY field that differs is `mcfm`.

## Batches

| batch | effect | LoRA trained on | arm | targets | prefix | shape | job ids |
|---|---|---|---|---|---|---|---|
| **B1** | lava | `spot_lava` | vanilla | horse, pumpkin | `r30_*` | pre-existing; 4 yaws each + pumpkin 360/720 | `2184849-2185000` |
| **B2** | lava | `spot_lava` | MCFM v2_D | horse, pumpkin | `r31_*` | 4 yaws each + pumpkin 360/720 | `2187321-2187330` |
| **B3** | rorschach (spot) | `spot_raurshaw` | vanilla + MCFM | monster, teapot, skull | `r32_*` | 2 arms x 3 targets x 4 yaws | `2187352-2187375` |
| **B4** | rorschach (spot) | `spot_raurshaw` | vanilla + MCFM | alien | `r32_*` | 2 arms x 4 yaws | `2187397-2187404` |
| **B5** | lava | `spot_lava` | vanilla + MCFM | hand | `r32_*` | 2 arms x 4 yaws | `2187435-2187442` |
| **B6** | rorschach (hand) + lava | `hand_rorschach / spot_lava` | MCFM v2_D only | penguin, mushroom, teapot | `r33_*` | 2 effects x 3 targets x 4 yaws | `2190550-2190573` |

| **B7** | **cross-effect** | `spot_lava` / `hand_rorschach` | MCFM v2_D only | penguin, mushroom | `r34_*` | 2 combos x 4 yaws | `2190589-2190596` |

### B7 is a different and stronger claim

B1-B6 hold the effect fixed and change the SHAPE: the adapter is always fed the same
video it was fitted against. B7 changes BOTH at once -- the conditioning is a video of an
effect the adapter never trained on, on a mesh it never saw:

| # | LoRA trained on | conditioning (unseen effect) | target mesh (unseen shape) | jobs |
|---|---|---|---|---|
| 1 | `spot_lava` | `data/hand_rorschach/frames_from_video` | penguin | `2190589,91,93,95` |
| 2 | `hand_rorschach` | `data/spot_lava/frames_from_video` | mushroom | `2190590,92,94,96` |

The two are deliberately crossed, so neither direction can be explained by one adapter
happening to be more permissive than the other.

If this works the adapter is not a lava edit or an ink edit -- it is a generic "lift
whatever this 2D video shows onto this 3D surface" operator, which is a far stronger
result than shape generalization alone.

WHERE IT COULD FAIL, so nobody over-reads a bad render: the adapter was fitted against
one effect's colour statistics. If the lava LoRA drags Rorschach ink toward orange, or the
ink LoRA flattens lava to black-and-white, that is the adapter having memorised an
appearance prior rather than a lifting operation. That outcome is still a publishable
finding -- it bounds what the method does -- but it is NOT the generalization claim, and
the figures must not be captioned as if it were.


| **B8** | rorschach + lava | `spot_raurshaw` / `skull_lava` | MCFM v2_D only | 11 fresh Dale meshes | `r35_*` | 2 effects x 11 meshes x 4 yaws | `2191140-2191227` |

### B8 uses raw meshes that had to be fitted first

data/_dale_meshes/*.obj are raw downloads: arbitrary centre, scale and up-axis. The
transfer renderer uses ONE fixed camera with world Z up, so a Y-up mesh renders on its
back and all four yaws are wrong together -- 4 wasted jobs per mesh before you find out.

`axis_sheet_cpu2.py` rasterises each mesh at six candidate frames through the identical
projection, `fit_dale.py` then bakes the chosen rotation + centre + unit scale into
`data/_dale_fitted/<name>.obj`. Vertex and face counts are asserted unchanged, so the
mesh in the figure is provably the mesh that was downloaded. Originals untouched.

| mesh | axis | verts / faces |
|---|---|---|
| dragons_xyzrgb_dragon | B_y2z | 124,943 / 249,882 |
| statues_napoleon | B_y2z | 49,501 / 98,998 |
| thingi10k_wooly_sheep | E_flipz | 42,101 / 84,198 |
| humanoid_mike_wazowski | B_y2z | 34,066 / 37,822 |
| aliens_ufo | E_flipz | 30,906 / 26,372 |
| furnature_chair | B_y2z | 21,896 / 43,796 |
| thingi10k_octocat | E_flipz | 18,944 / 37,884 |
| vehicles_tie_fighter | B_y2z | 16,820 / 33,154 |
| animals_blub | B_y2z | 7,317 / 14,208 |
| animals_bob | B_y2z | 5,647 / 10,688 |
| animals_octopus | B_y2z | 2,827 / 5,215 |

Axis choices come from `data/_dale_meshes/axis/*.png`; re-check there before adding meshes.


| **B8v2** | rorschach + lava | `spot_raurshaw` / `skull_lava` | MCFM v2_D only | 11 Dale meshes, CORRECTED orientation | `r36_*` | 2 x 11 x 4 | `2192806-2192894` |

### B8 v1 was inverted -- what went wrong and what stops it recurring

`r35_*` rendered every mesh upside down and was deleted. Cause: `axis_sheet_cpu2.py`
mapped NDC y with an extra `(1 - ...)` flip, so its sheets were a VERTICAL MIRROR of what
nvdiffrast produces. Every axis chosen from them was therefore inverted.

The coverage gate in place at the time could not catch this -- an upside-down silhouette
covers exactly the same pixel count. **GATE-updir** replaces it: it projects the highest
world-Z vertex of a known mesh and asserts the result lands in the top half of the image.
That fails loudly on a flip.

Confirmed against ground truth before changing anything: world +Z projects to NEGATIVE
ndc_y, and nvdiffrast row 0 is the top of the image, so no extra flip belongs there.

Corrected axes (all differ from v1):

| mesh | axis |
|---|---|
| dragons_xyzrgb_dragon | `C_z2y` |
| statues_napoleon | `C_z2y` |
| thingi10k_wooly_sheep | `A_xyz` |
| humanoid_mike_wazowski | `C_z2y` |
| aliens_ufo | `C_z2y` |
| furnature_chair | `C_z2y` |
| thingi10k_octocat | `A_xyz` |
| vehicles_tie_fighter | `C_z2y` |
| animals_blub | `C_z2y` |
| animals_bob | `C_z2y` |
| animals_octopus | `C_z2y` |


## Target meshes

| target | mesh | verts / faces |
|---|---|---|
| horse | `data/horse_metal/mesh/horse_metal_render_frame_guan.obj` | 1,498 / 1,700 |
| pumpkin | `data/pumpkin_rot/mesh/pumpkin_rot_render_frame_guan.obj` | 1,616 / 1,700 |
| hand | `data/hand_rorschach/mesh/hand_rorschach_render_frame.obj` | 9,119 / 15,982 |
| monster | `data/monster_lava_2/mesh/monster_lava_2_render_frame.obj` | 17,514 / 29,170 |
| teapot | `data/teapot_ceramic_crack/mesh/teapot_ceramic_crack_render_frame_guan.obj` | 8,334 / 15,704 |
| skull | `data/skull_lava/mesh/skull_lava_render_frame.obj` | 30,936 / 10,312 |
| alien | `data/alien_glow/mesh/alien_glow_render_frame.obj` | 34,217 / 68,430 |
| penguin | `data/penguin_circuits/mesh/penguin_circuits_render_frame_guan.obj` | 1,679 / 1,900 |
| mushroom | `data/mushroom_glow/mesh/mushroom_glow_render_frame.obj` (symlink to `dynamesh_meshes/hero/mushroom_hero.obj`; hero mesh, no GT video) | 69,492 / 23,166 |

## B6 status (the batch still in flight)

| tag | job | state | frames |
|---|---|---|---|
| `r33_mcfm_ror_penguin_yaw000` | 2190550 | PENDING | 0 |
| `r33_mcfm_ror_penguin_yaw090` | 2190551 | PENDING | 0 |
| `r33_mcfm_ror_penguin_yaw180` | 2190552 | PENDING | 0 |
| `r33_mcfm_ror_penguin_yaw270` | 2190553 | PENDING | 0 |
| `r33_mcfm_ror_mushroom_yaw000` | 2190554 | PENDING | 0 |
| `r33_mcfm_ror_mushroom_yaw090` | 2190555 | PENDING | 0 |
| `r33_mcfm_ror_mushroom_yaw180` | 2190556 | PENDING | 0 |
| `r33_mcfm_ror_mushroom_yaw270` | 2190557 | PENDING | 0 |
| `r33_mcfm_ror_teapot_yaw000` | 2190558 | PENDING | 0 |
| `r33_mcfm_ror_teapot_yaw090` | 2190559 | PENDING | 0 |
| `r33_mcfm_ror_teapot_yaw180` | 2190560 | PENDING | 0 |
| `r33_mcfm_ror_teapot_yaw270` | 2190561 | PENDING | 0 |
| `r33_mcfm_lava_penguin_yaw000` | 2190562 | PENDING | 0 |
| `r33_mcfm_lava_penguin_yaw090` | 2190563 | PENDING | 0 |
| `r33_mcfm_lava_penguin_yaw180` | 2190564 | PENDING | 0 |
| `r33_mcfm_lava_penguin_yaw270` | 2190565 | PENDING | 0 |
| `r33_mcfm_lava_mushroom_yaw000` | 2190566 | PENDING | 0 |
| `r33_mcfm_lava_mushroom_yaw090` | 2190567 | PENDING | 0 |
| `r33_mcfm_lava_mushroom_yaw180` | 2190568 | PENDING | 0 |
| `r33_mcfm_lava_mushroom_yaw270` | 2190569 | PENDING | 0 |
| `r33_mcfm_lava_teapot_yaw000` | 2190570 | PENDING | 0 |
| `r33_mcfm_lava_teapot_yaw090` | 2190571 | PENDING | 0 |
| `r33_mcfm_lava_teapot_yaw180` | 2190572 | PENDING | 0 |
| `r33_mcfm_lava_teapot_yaw270` | 2190573 | PENDING | 0 |

## Not valid as targets

The spot mesh (`spot`, `spot_lava`, `spot_raurshaw`, `spot_star` -- all identical geometry)
is the SOURCE shape for the lava and spot-rorschach adapters. GATE-transfer refuses it.
Likewise `hand_rorschach` is the source shape for the hand-rorschach adapter, so it can only
be a target for the lava effect -- which is exactly what B5 does.

## Unused meshes still available (deduped by geometry)

`ancient_lady` 39,349v/70,000f | `lionstatue` 36,625v/63,338f | `eagle_blackness` 32,777v/51,482f |
`sword` 24,271v/44,582f | `whale_spots` 4,114v/5,000f | `wingnut` 3,463v/6,090f

`sword` and `wingnut` are the stress cases -- everything transferred so far is a closed blobby
volume, whereas a sword is thin and near-planar and a wingnut has a hole and machined edges.

## Artifact

Batches B1-B5 are published with all 60 videos embedded:
https://claude.ai/code/artifact/add29505-9fdf-483a-a44d-b5ad7ef6fda7

Rebuild after new renders land:
```
/tmp/prep_vids.sh                 # re-encode to 720px/crf30 for embedding
python /tmp/gen_manifest.py       # re-probe every path from disk
python /tmp/gen_html.py           # rebuild the page
```
