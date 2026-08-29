# A + B Results Handoff — file paths for the paper writeup

For Guan. One section per object. Every path below is on disk as of
2026-08-23 01:55 CDT. Paths not yet on disk (grid panel still building) are
marked PENDING, not omitted, so nothing has to be re-derived later.

Convention: `run27` = baseline (cross+self LoRA); `+MCFM` = temporal-only
conditioning added; `run28/29/30` = KL variants; `run31/33` = dual branch;
`run32/34` = wide context (no MCFM — mutually exclusive by design).

## `ancient_lady_effect_2`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/ancient_lady_effect_2/video/ancient_lady_effect_2.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/ancient_lady_effect_2/mesh/ancient_lady_effect_2_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/ancient_lady_effect_2/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_ancient_lady_effect_2/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e05990f9` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e05990f9/ckpts/lora_best.pt` | 28.801 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_e2b7999b` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_e2b7999b/ckpts/lora_best.pt` | 28.743 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_58197237` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_58197237/ckpts/lora_best.pt` | 28.539 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_9e919330` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_9e919330/ckpts/lora_best.pt` | 28.461 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_0b875b8f` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_0b875b8f/ckpts/lora_best.pt` | 28.609 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_4b802338` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_4b802338/ckpts/lora_best.pt` | 28.545 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_5d62a672` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_5d62a672/ckpts/lora_best.pt` | 28.383 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d51276ec` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d51276ec/ckpts/lora_best.pt` | 28.313 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_91cc3ce3` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_91cc3ce3/ckpts/lora_best.pt` | 28.841 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_e3688e17` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_e3688e17/ckpts/lora_best.pt` | 28.818 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_293bacb9` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_293bacb9/ckpts/lora_best.pt` | 28.755 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_0ef118dc` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_0ef118dc/ckpts/lora_best.pt` | 28.388 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_6ab71641` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_6ab71641/ckpts/lora_best.pt` | 28.480 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_bcd07020` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_bcd07020/ckpts/lora_best.pt` | 28.414 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_ancient_lady_effect_2_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_ancient_lady_effect_2_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_ancient_lady_effect_2_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_ancient_lady_effect_2_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/ancient_lady_effect_2_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/ancient_lady_effect_2_diagA_grid.mp4`
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/ancient_lady_effect_2_diagB_grid.mp4`
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/ancient_lady_effect_2_diagC_grid.mp4`  **PENDING** (building now)

---

## `spot_lava`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/video/spot_lava.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/mesh/spot_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_spot_lava/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_a032b897` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_a032b897/ckpts/lora_best.pt` | 19.731 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_9d487032` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_9d487032/ckpts/lora_best.pt` | 20.210 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_46e7e464` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_46e7e464/ckpts/lora_best.pt` | 23.778 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_be99d947` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_be99d947/ckpts/lora_best.pt` | 23.840 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_4b2136a3` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_4b2136a3/ckpts/lora_best.pt` | 23.691 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_65298a88` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_65298a88/ckpts/lora_best.pt` | 23.900 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_54ce37d8` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_54ce37d8/ckpts/lora_best.pt` | 23.491 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8db35a31` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8db35a31/ckpts/lora_best.pt` | 23.118 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_54d22a32` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_54d22a32/ckpts/lora_best.pt` | 24.166 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_c60dddfd` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_c60dddfd/ckpts/lora_best.pt` | 24.315 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_bc3623c7` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_bc3623c7/ckpts/lora_best.pt` | 24.102 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_23515ead` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_23515ead/ckpts/lora_best.pt` | 23.614 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_8313091c` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_8313091c/ckpts/lora_best.pt` | 23.626 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_f65db5f8` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_f65db5f8/ckpts/lora_best.pt` | 23.402 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_spot_lava_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_spot_lava_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_spot_lava_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_spot_lava_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/spot_lava_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/spot_lava_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/spot_lava_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/spot_lava_diagC_grid.mp4`  **PENDING** (building now)

---

## `pumpkin_rot`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/pumpkin_rot/video/kling_pumpkin_rot.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/pumpkin_rot/mesh/pumpkin_rot_render_frame_guan.obj`
- **GT frames** (121): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/pumpkin_rot/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_pumpkin_rot/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_f9e70449` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_f9e70449/ckpts/lora_best.pt` | 33.723 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_a08b9a37` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_a08b9a37/ckpts/lora_best.pt` | 33.816 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_e1676da2` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_e1676da2/ckpts/lora_best.pt` | 33.200 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_38024165` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_38024165/ckpts/lora_best.pt` | 33.318 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_752d0bc6` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_752d0bc6/ckpts/lora_best.pt` | 33.429 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1953339b` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1953339b/ckpts/lora_best.pt` | 33.369 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_aa4cabb8` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_aa4cabb8/ckpts/lora_best.pt` | 32.852 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_b2024c04` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_b2024c04/ckpts/lora_best.pt` | 32.832 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_20d41866` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_20d41866/ckpts/lora_best.pt` | 33.793 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_93b60206` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_93b60206/ckpts/lora_best.pt` | 33.885 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_aab9e545` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_aab9e545/ckpts/lora_best.pt` | 33.776 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_896e2ced` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_896e2ced/ckpts/lora_best.pt` | 32.864 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_ca87e680` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_ca87e680/ckpts/lora_best.pt` | 32.967 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e9dbe316` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e9dbe316/ckpts/lora_best.pt` | 32.909 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_pumpkin_rot_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_pumpkin_rot_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_pumpkin_rot_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_pumpkin_rot_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/pumpkin_rot_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/pumpkin_rot_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/pumpkin_rot_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/pumpkin_rot_diagC_grid.mp4`  **PENDING** (building now)

---

## `hand_rorschach`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/hand_rorschach/video/hand_rorschach.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/hand_rorschach/mesh/hand_rorschach_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/hand_rorschach/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_hand_rorschach/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_1eef6f88` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_1eef6f88/ckpts/lora_best.pt` | 24.831 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_eaa1fb67` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_eaa1fb67/ckpts/lora_best.pt` | 25.057 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_53ff6c61` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_53ff6c61/ckpts/lora_best.pt` | 24.376 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_b8cf386f` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_b8cf386f/ckpts/lora_best.pt` | 24.462 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_d3807969` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_d3807969/ckpts/lora_best.pt` | 24.633 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_64047ce7` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_64047ce7/ckpts/lora_best.pt` | 24.671 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_2fba9a5e` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_2fba9a5e/ckpts/lora_best.pt` | 24.064 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_97289bf2` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_97289bf2/ckpts/lora_best.pt` | 24.247 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_b3ff4917` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_b3ff4917/ckpts/lora_best.pt` | 24.956 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_80d3d567` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_80d3d567/ckpts/lora_best.pt` | 24.942 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_ddaa6fcb` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_ddaa6fcb/ckpts/lora_best.pt` | 25.015 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a5b7a838` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a5b7a838/ckpts/lora_best.pt` | 24.279 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_2089b540` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_2089b540/ckpts/lora_best.pt` | 24.376 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_744eb2e4` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_744eb2e4/ckpts/lora_best.pt` | 24.006 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_hand_rorschach_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_hand_rorschach_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_hand_rorschach_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_hand_rorschach_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/hand_rorschach_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/hand_rorschach_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/hand_rorschach_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/hand_rorschach_diagC_grid.mp4`  **PENDING** (building now)

---

## `animal_blob_crack`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_crack/video/animal_blob_crack.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_crack/mesh/animal_blob_crack_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_crack/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_animal_blob_crack/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_ab2f37ce` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_ab2f37ce/ckpts/lora_best.pt` | 25.144 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_242df884` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_242df884/ckpts/lora_best.pt` | 25.275 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_f2b6b779` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_f2b6b779/ckpts/lora_best.pt` | 24.750 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_fd58cabb` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_fd58cabb/ckpts/lora_best.pt` | 24.913 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_706406c6` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_706406c6/ckpts/lora_best.pt` | 24.853 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_678841ed` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_678841ed/ckpts/lora_best.pt` | 24.839 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_6bf2e4ed` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_6bf2e4ed/ckpts/lora_best.pt` | 24.667 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_c65c6272` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_c65c6272/ckpts/lora_best.pt` | 24.550 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_ebac0e47` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_ebac0e47/ckpts/lora_best.pt` | 25.308 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_0a864f0f` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_0a864f0f/ckpts/lora_best.pt` | 25.289 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_34834fab` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_34834fab/ckpts/lora_best.pt` | 25.320 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a2f67d3b` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a2f67d3b/ckpts/lora_best.pt` | 24.597 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_5cc4cce0` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_5cc4cce0/ckpts/lora_best.pt` | 24.769 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_0967e287` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_0967e287/ckpts/lora_best.pt` | 24.682 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_animal_blob_crack_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_animal_blob_crack_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_animal_blob_crack_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_animal_blob_crack_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_crack_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_crack_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_crack_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_crack_diagC_grid.mp4`  **PENDING** (building now)

---

## `animal_blob_orange_crack`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_orange_crack/video/animal_blob_orange_crack.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_orange_crack/mesh/animal_blob_orange_crack_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/animal_blob_orange_crack/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_animal_blob_orange_crack/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_f2aada81` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_f2aada81/ckpts/lora_best.pt` | 27.456 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_3619d303` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_3619d303/ckpts/lora_best.pt` | 27.763 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_ec70bedf` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_ec70bedf/ckpts/lora_best.pt` | 27.225 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_f107c540` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_f107c540/ckpts/lora_best.pt` | 27.405 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_dbdcf3eb` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_dbdcf3eb/ckpts/lora_best.pt` | 27.363 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_ffa8ee00` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_ffa8ee00/ckpts/lora_best.pt` | 27.421 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_70292c1d` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_70292c1d/ckpts/lora_best.pt` | 27.048 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8f720661` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8f720661/ckpts/lora_best.pt` | 27.060 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_7ace12e0` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_7ace12e0/ckpts/lora_best.pt` | 27.644 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_3c7cb318` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_3c7cb318/ckpts/lora_best.pt` | 27.606 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_f9be6d5a` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_f9be6d5a/ckpts/lora_best.pt` | 27.324 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_7fa86548` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_7fa86548/ckpts/lora_best.pt` | 27.083 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_80b828ea` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_80b828ea/ckpts/lora_best.pt` | 27.287 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e24ab01f` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e24ab01f/ckpts/lora_best.pt` | 26.917 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_animal_blob_orange_crack_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_animal_blob_orange_crack_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_animal_blob_orange_crack_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_animal_blob_orange_crack_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_orange_crack_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_orange_crack_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_orange_crack_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/animal_blob_orange_crack_diagC_grid.mp4`  **PENDING** (building now)

---

## `chair_real_wooden_crack`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_real_wooden_crack/video/chair_real_wooden_crack.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_real_wooden_crack/mesh/chair_real_wooden_crack_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_real_wooden_crack/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_chair_real_wooden_crack/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_dc5331ae` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_dc5331ae/ckpts/lora_best.pt` | 25.573 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_55e4f367` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_55e4f367/ckpts/lora_best.pt` | 25.310 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_9558dd9c` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_9558dd9c/ckpts/lora_best.pt` | 25.169 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_a1f37adc` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_a1f37adc/ckpts/lora_best.pt` | 25.135 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_d7ac67c9` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_d7ac67c9/ckpts/lora_best.pt` | 25.256 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_fd11f833` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_fd11f833/ckpts/lora_best.pt` | 25.193 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_881b1c6b` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_881b1c6b/ckpts/lora_best.pt` | 24.955 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_e72ecf58` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_e72ecf58/ckpts/lora_best.pt` | 24.897 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_ad6a9c57` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_ad6a9c57/ckpts/lora_best.pt` | 25.573 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_9c0d29a5` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_9c0d29a5/ckpts/lora_best.pt` | 25.588 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_d9db36c1` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_d9db36c1/ckpts/lora_best.pt` | 25.361 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a33041f5` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_a33041f5/ckpts/lora_best.pt` | 24.952 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_21bcd65b` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_21bcd65b/ckpts/lora_best.pt` | 24.868 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_8c4c8582` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_8c4c8582/ckpts/lora_best.pt` | 24.845 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_chair_real_wooden_crack_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_chair_real_wooden_crack_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_chair_real_wooden_crack_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_chair_real_wooden_crack_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_real_wooden_crack_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_real_wooden_crack_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_real_wooden_crack_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_real_wooden_crack_diagC_grid.mp4`  **PENDING** (building now)

---

## `chair_ice`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_ice/video/chair_ice.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_ice/mesh/chair_ice_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_ice/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_chair_ice/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_ae22fe84` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_ae22fe84/ckpts/lora_best.pt` | 21.370 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d103d91e` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d103d91e/ckpts/lora_best.pt` | 21.562 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_db5a6b8b` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_db5a6b8b/ckpts/lora_best.pt` | 21.292 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_df69b17c` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_df69b17c/ckpts/lora_best.pt` | 21.322 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_bf03b767` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_bf03b767/ckpts/lora_best.pt` | 21.262 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8781065b` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8781065b/ckpts/lora_best.pt` | 21.204 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_87c71ba9` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_87c71ba9/ckpts/lora_best.pt` | 21.127 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_74b84809` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_74b84809/ckpts/lora_best.pt` | 20.979 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_c5247ef3` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_c5247ef3/ckpts/lora_best.pt` | 21.587 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_1022a7d5` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_1022a7d5/ckpts/lora_best.pt` | 21.676 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_5c69c502` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_5c69c502/ckpts/lora_best.pt` | 21.531 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_02b53320` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_02b53320/ckpts/lora_best.pt` | 21.064 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_eaca302d` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_eaca302d/ckpts/lora_best.pt` | 21.330 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e0861903` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_e0861903/ckpts/lora_best.pt` | 21.041 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_chair_ice_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_chair_ice_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_chair_ice_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_chair_ice_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_ice_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_ice_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_ice_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_ice_diagC_grid.mp4`  **PENDING** (building now)

---

## `chair_moss`

- **video**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_moss/video/chair_moss.mp4`
- **mesh**: `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_moss/mesh/chair_moss_render_frame.obj`
- **GT frames** (150): `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_moss/frames_from_video/`
- **2D-copy targets**: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_chair_moss/frames/`

| rung | run dir | checkpoint | PSNR |
|---|---|---|---|
| 27 | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_a33d1de5` | `runs/rung27_l1_lp_all_qkvo+sa_r4_s42_a33d1de5/ckpts/lora_best.pt` | 27.913 |
| 27 +MCFM | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8cd9be64` | `runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8cd9be64/ckpts/lora_best.pt` | 27.999 |
| 28 | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_c1669502` | `runs/rung28_l1_lp_all_qkvo+sa_r4_s42_c1669502/ckpts/lora_best.pt` | 27.508 |
| 28 +MCFM | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d071741c` | `runs/rung28_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_d071741c/ckpts/lora_best.pt` | 27.447 |
| 29 | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_4380cde5` | `runs/rung29_l1_lp_all_qkvo+sa_r4_s42_4380cde5/ckpts/lora_best.pt` | 27.613 |
| 29 +MCFM | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_7a0cfa23` | `runs/rung29_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_7a0cfa23/ckpts/lora_best.pt` | 27.711 |
| 30 | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_6e978a06` | `runs/rung30_l1_lp_all_qkvo+sa_r4_s42_6e978a06/ckpts/lora_best.pt` | 26.904 |
| 30 +MCFM | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1b1a802e` | `runs/rung30_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1b1a802e/ckpts/lora_best.pt` | 27.122 |
| 31 | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_dd65c3f5` | `runs/rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42_dd65c3f5/ckpts/lora_best.pt` | 28.020 |
| 31 +MCFM | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_02decb95` | `runs/rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_02decb95/ckpts/lora_best.pt` | 28.074 |
| 32 | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_c52c0820` | `runs/rung32_l1_lp_cw3_all_qkvo+sa_r4_s42_c52c0820/ckpts/lora_best.pt` | 27.840 |
| 33 | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_0843baa8` | `runs/rung33_l1_lp_tw3b_all_qkvo+sa_r4_s42_0843baa8/ckpts/lora_best.pt` | 27.067 |
| 33 +MCFM | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_1a6d8513` | `runs/rung33_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42_1a6d8513/ckpts/lora_best.pt` | 26.971 |
| 34 | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_4cb84c12` | `runs/rung34_l1_lp_cw3_all_qkvo+sa_r4_s42_4cb84c12/ckpts/lora_best.pt` | 27.080 |

**Renders** (fixed camera, texture-only motion — required for flicker):

| view | camera | frames dir pattern |
|---|---|---|
| train | training view (yaw0, elev0) | `out/view_chair_moss_<arm>_train/frames/` (arm = 27,27m,28,28m,...,34) |
| diagA | yaw 45°, elev +25° | `out/view_chair_moss_<arm>_diagA/frames/` (arm = 27,27m,28,28m,...,34) |
| diagB | yaw 135°, elev −20° | `out/view_chair_moss_<arm>_diagB/frames/` (arm = 27,27m,28,28m,...,34) |
| diagC | yaw 225°, elev +30° | `out/view_chair_moss_<arm>_diagC/frames/` (arm = 27,27m,28,28m,...,34) |

**Grid panel (GT + frozen + 14 arms, one video per view):**

- train: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_moss_train_grid.mp4`  **PENDING** (building now)
- diagA: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_moss_diagA_grid.mp4`  **PENDING** (building now)
- diagB: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_moss_diagB_grid.mp4`  **PENDING** (building now)
- diagC: `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_moss_diagC_grid.mp4`  **PENDING** (building now)

---
