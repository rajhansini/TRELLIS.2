# Render paths — dynamic texture, 13 objects

Base: `/net/projects/ranalab/rajhansini/TRELLIS.2`  (host `fe02.ai.cs.uchicago.edu`)

Everything below is on the cluster. Nothing needs our training pipeline to view.

---

## THE ONE THING TO KNOW FIRST

There is **no baked UV texture map**. The texture is produced by TRELLIS.2 with a LoRA
applied at render time, so you cannot load a textured mesh in Blender and light it.
Work from the rendered frames below, or tell us a camera and we render it.

---

## Camera (identical everywhere in this project)

```
EXTRINSICS = [[1,0,0,0],[0,0,-1,0],[0,1,0,2],[0,0,0,1]]   # eye (0,-2,0), +z up
fov        = 40 deg      (fx = fy = 1/(2*tan(20deg)))
near, far  = 0.5, 3.0
res        = 960 for targets, 518 for the view renders, 1024 for the spot_lava figure set
```

Orbits use yaw about the world vertical, elevation above the horizon, radius 2.
yaw 0 / elev 0 is the training view.

---

## The six cameras, per object

| view | camera | fixed? |
|---|---|---|
| `train` | yaw 0, elev 0, fixed — the training view | yes |
| `orbit360` | one revolution over the sequence, elev 15 | no — turning |
| `orbit720` | two revolutions, elev 15 | no — turning |
| `diagA` | yaw 45, elev +25, fixed | yes |
| `diagB` | yaw 135, elev -20, fixed | yes |
| `diagC` | yaw 225, elev +30, fixed | yes |

---

## Per-object paths

### mushroom_glow

mesh `mushroom` &middot; 150 frames &middot; rung27 **29.138** dB -> +MCFM **29.297** dB

```
video      data/mushroom_glow/video/mushroom_glow.mp4
frames     data/mushroom_glow/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/mushroom_glow/mesh/mushroom_glow_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_mushroom_glow/frames/
train      experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_mushroom_glow_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/mushroom_glow_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### monster_lava_2

mesh `armadillo` &middot; 150 frames &middot; rung27 **26.905** dB -> +MCFM **26.951** dB

```
video      data/monster_lava_2/video/monster_lava_2.mp4
frames     data/monster_lava_2/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/monster_lava_2/mesh/monster_lava_2_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_monster_lava_2/frames/
train      experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_monster_lava_2_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/monster_lava_2_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### teapot_crack

mesh `teapot_ceramic_crack` &middot; 150 frames &middot; rung27 **26.508** dB -> +MCFM **26.419** dB

```
video      data/teapot_crack/video/teapot_crack.mp4
frames     data/teapot_crack/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/teapot_crack/mesh/teapot_crack_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_teapot_crack/frames/
train      experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_teapot_crack_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/teapot_crack_<view>_GT_frozen_r27_mcfm.mp4  [6/6 built]
```

### ancient_statue_clay

mesh `nefertiti` &middot; 150 frames &middot; rung27 **25.648** dB -> +MCFM **25.697** dB

```
video      data/ancient_statue_clay/video/ancient_statue_clay.mp4
frames     data/ancient_statue_clay/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/ancient_statue_clay/mesh/ancient_statue_clay_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_ancient_statue_clay/frames/
train      experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_ancient_statue_clay_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/ancient_statue_clay_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### skull_lava

mesh `skull` &middot; 150 frames &middot; rung27 **25.347** dB -> +MCFM **25.685** dB

```
video      data/skull_lava/video/skull_lava.mp4
frames     data/skull_lava/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/skull_lava/mesh/skull_lava_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_skull_lava/frames/
train      experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_skull_lava_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/skull_lava_<view>_GT_frozen_r27_mcfm.mp4  [6/6 built]
```

### hand_rorschach

mesh `hand` &middot; 150 frames &middot; rung27 **24.831** dB -> +MCFM **25.057** dB

```
video      data/hand_rorschach/video/hand_rorschach.mp4
frames     data/hand_rorschach/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/hand_rorschach/mesh/hand_rorschach_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_hand_rorschach/frames/
train      experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_hand_rorschach_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/hand_rorschach_<view>_GT_frozen_r27_mcfm.mp4  [6/6 built]
```

### monster_rainbow

mesh `armadillo` &middot; 150 frames &middot; rung27 **24.376** dB -> +MCFM **24.466** dB

```
video      data/monster_rainbow/video/monster_rainbow.mp4
frames     data/monster_rainbow/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/monster_rainbow/mesh/monster_rainbow_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_monster_rainbow/frames/
train      experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_monster_rainbow_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/monster_rainbow_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### ancient_lady

mesh `nefertiti` &middot; 135 frames &middot; rung27 **23.492** dB -> +MCFM **23.493** dB

```
video      data/ancient_lady/video/ancient_lady.mp4
frames     data/ancient_lady/frames_from_video/frame_0001.png .. frame_0135.png
mesh       data/ancient_lady/mesh/ancient_lady_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_ancient_lady/frames/
train      experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_ancient_lady_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/ancient_lady_<view>_GT_frozen_r27_mcfm.mp4  [6/6 built]
```

### eagle_blackness

mesh `falconstatue` &middot; 150 frames &middot; rung27 **23.058** dB -> +MCFM **23.064** dB

```
video      data/eagle_blackness/video/eagle_blackness.mp4
frames     data/eagle_blackness/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/eagle_blackness/mesh/eagle_blackness_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_eagle_blackness/frames/
train      experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_eagle_blackness_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/eagle_blackness_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### plane_waves

mesh `whale_spots` &middot; 150 frames &middot; rung27 **17.941** dB -> +MCFM **18.126** dB

```
video      data/plane_waves/video/plane_waves.mp4
frames     data/plane_waves/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/plane_waves/mesh/plane_waves_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_plane_waves/frames/
train      experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_train/frames/   [READY]
orbit360   experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_orbit360/frames/   [READY]
orbit720   experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_orbit720/frames/   [READY]
diagA      experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_diagA/frames/   [READY]
diagB      experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_diagB/frames/   [READY]
diagC      experiments/dynamesh/out/view_plane_waves_{r27,mcfm}_diagC/frames/   [READY]
panels     experiments/dynamesh/out/VIEWS/plane_waves_<view>_GT_frozen_r27_mcfm.mp4  [6/6 built]
```

### alien_glow

mesh `alien` &middot; 150 frames &middot; _training still running_

```
video      data/alien_glow/video/alien_glow.mp4
frames     data/alien_glow/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/alien_glow/mesh/alien_glow_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_alien_glow/frames/
train      experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_train/frames/   [rendering 0/0]
orbit360   experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_orbit360/frames/   [rendering 0/0]
orbit720   experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_orbit720/frames/   [rendering 0/0]
diagA      experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_diagA/frames/   [rendering 0/0]
diagB      experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_diagB/frames/   [rendering 0/0]
diagC      experiments/dynamesh/out/view_alien_glow_{r27,mcfm}_diagC/frames/   [rendering 0/0]
panels     experiments/dynamesh/out/VIEWS/alien_glow_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### ancient_lady_crack

mesh `nefertiti` &middot; 150 frames &middot; _training still running_

```
video      data/ancient_lady_crack/video/ancient_lady_crack.mp4
frames     data/ancient_lady_crack/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/ancient_lady_crack/mesh/ancient_lady_crack_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_ancient_lady_crack/frames/
train      experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_train/frames/   [rendering 0/0]
orbit360   experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_orbit360/frames/   [rendering 0/0]
orbit720   experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_orbit720/frames/   [rendering 0/0]
diagA      experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_diagA/frames/   [rendering 0/0]
diagB      experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_diagB/frames/   [rendering 0/0]
diagC      experiments/dynamesh/out/view_ancient_lady_crack_{r27,mcfm}_diagC/frames/   [rendering 0/0]
panels     experiments/dynamesh/out/VIEWS/ancient_lady_crack_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

### spot_raurshaw

mesh `spot_lava` &middot; 150 frames &middot; _training still running_

```
video      data/spot_raurshaw/video/spot_raurshaw.mp4
frames     data/spot_raurshaw/frames_from_video/frame_0001.png .. frame_0150.png
mesh       data/spot_raurshaw/mesh/spot_raurshaw_render_frame.obj      (already in the camera frame)
targets    experiments/dynamesh/out/gt_targets_spot_raurshaw/frames/
train      experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_train/frames/   [rendering 0/0]
orbit360   experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_orbit360/frames/   [rendering 0/0]
orbit720   experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_orbit720/frames/   [rendering 0/0]
diagA      experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_diagA/frames/   [rendering 0/0]
diagB      experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_diagB/frames/   [rendering 0/0]
diagC      experiments/dynamesh/out/view_spot_raurshaw_{r27,mcfm}_diagC/frames/   [rendering 0/0]
panels     experiments/dynamesh/out/VIEWS/spot_raurshaw_<view>_GT_frozen_r27_mcfm.mp4  [0/6 built]
```

---

## What the files are

- **`view_<obj>_<arm>_<view>/frames/NNNN.png`** — each image is `frozen | adapted` **side by side**.
  Left half = frozen TRELLIS.2, right half = the trained arm.
  `r27` = rank-4 LoRA on cross-attention AND self-attention. `mcfm` = same plus temporal
  blending of the cached DINOv3 conditioning (v2_D).
- **`out/VIEWS/<obj>_<view>_GT_frozen_r27_mcfm.mp4`** — 4-panel video, already composited:
  ground truth | frozen | rung27 | rung27+MCFM.
- **`out/gt_targets_<obj>/frames/`** — the 2D copy: the video's colour sampled inside our own
  rasterised silhouette, thin rim filled from the nearest valid neighbour. **This, not the raw
  video, is what the PSNR numbers are measured against** — if a figure shows a "ground truth"
  column next to a PSNR, it should be these.

---

## Splitting a side-by-side render

```bash
# right half = our result;  top 28 px is a text bar, cropped off
ffmpeg -i frames/0112.png -vf "crop=iw/2:ih-28:iw/2:28" ours.png
ffmpeg -i frames/0112.png -vf "crop=iw/2:ih-28:0:28"    frozen.png

# a video of just our result
ffmpeg -framerate 20 -start_number 1 -i frames/%04d.png \
  -vf "crop=iw/2:ih-28:iw/2:28" -c:v libx264 -crf 18 -pix_fmt yuv420p ours.mp4
```

Frame 112 is the one we have been using for stills — the effect is fully developed but not
yet at its most saturated. Frames 1-20 are still mostly untextured.

---

## spot_lava figure set (1024px, for the system figure)

```
bundle   handoff/spot_lava_r27_mcfm/          (mesh, video, frames, targets, ckpts, README)
1024 px  experiments/dynamesh/out/guan_spot_lava_mcfm_<view>/frames/
```

Same six views. 24.194 dB / 0.8347 SSIM, against 9.446 / 0.2670 frozen.

---

## Copying it down

```bash
cd ~/Downloads && scp -r rajhansini@fe02.ai.cs.uchicago.edu:\
  /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/VIEWS .
```

That is the composited 4-panel videos only, which is usually what you want. Add
`out/view_*` for the raw frames, or `handoff/spot_lava_r27_mcfm` for the figure bundle.

---

## Stating MCFM

Its PSNR gain is small (mean +0.10 dB over 10 objects) and a reviewer will read that as noise.
The defensible claim is temporal: measured as flicker (second temporal difference) it is
**-38.5% at the training view, -41.5% on unseen views, lower on 32 of 32 cells** across the
earlier eight-object set. Quote it that way.

