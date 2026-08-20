# spot_lava — rung27 + MCFM  (handoff for the system figure)

Everything needed to make figure visuals for the spot lava result, without
running our training pipeline.

**Result: 24.194 dB PSNR / 0.8347 SSIM**, against **9.446 dB / 0.2670** for frozen
TRELLIS.2 on the same mesh, same targets, same 150 frames.

---

## What is in here

| folder | what it is |
|---|---|
| `mesh/spot_render_frame.obj` | **the mesh to render.** Already sits in the pipeline camera frame — do not rotate, centre or rescale it. |
| `mesh/spot.obj` | the raw mesh before that pose was baked in. For reference only. |
| `video/spot_lava.mp4` | the Kling clip. Single fixed camera, 150 frames. |
| `video_frames/frame_0001..0150.png` | that clip as frames, 1440². This is "the current lava video". |
| `targets_2d_copy/gt_*.png` | what we actually train against, 960². See below. |
| `renders_518_4azimuths/yaw{0,90,180,270}/` | our renders, 150 frames each, 518². **Each image is `frozen | adapted` side by side** — left half frozen TRELLIS.2, right half rung27+MCFM. |
| `checkpoints/lora_best_r27_mcfm_v2D.pt` | the rank-4 LoRA for the result above. |
| `checkpoints/lora_best_r27_plain.pt` | same but without MCFM, for the ablation. |
| `config/` | exact training config and per-frame eval for both arms. |

Higher-resolution renders (1024², training view + 360 + 720 + three diagonal
views) are rendering now and will be added as `renders_1024/`.

---

## The camera — this is the part that matters

Every render in this project uses ONE camera, and the mesh is already posed for
it. In our convention:

```
EXTRINSICS = [[1, 0,  0, 0],
              [0, 0, -1, 0],
              [0, 1,  0, 2],
              [0, 0,  0, 1]]        # eye at (0,-2,0), +z is up
fov        = 40°   (so fx = fy = 1 / (2·tan(20°)))
near, far  = 0.5, 3.0
resolution = 960 for targets, 518 or 1024 for renders
```

Equivalently: the camera sits 2 units out along −y looking at the origin, with
+z up. `mesh/spot_render_frame.obj` is already in that frame, so rendering it
with this camera reproduces the view the Kling clip was generated from.

If you orbit, we use yaw about the world vertical and elevation above the
horizon, radius 2. yaw 0 / elev 0 is the training view.

---

## What the "targets" are, and why they are not the video

`targets_2d_copy/` is **not** the Kling video. For each frame we rasterise our
mesh's silhouette and copy the video's colour at those same pixels; the thin rim
where the two silhouettes disagree is filled from the nearest valid neighbour.
So the target is the video's texture restricted to our geometry.

For spot_lava the two silhouettes agree closely, so the targets are ~99% straight
copy. They are what the numbers above are measured against, so if the figure shows
a "ground truth" column next to a PSNR, that column should be these, not the raw
video frames.

The video is still the right thing to show as "input" — just don't label it as
what the loss saw.

---

## Making a still or a video from the renders as-is

Each render frame is a side-by-side pair. To split:

```bash
# right half = our result (rung27 + MCFM)
ffmpeg -i renders_518_4azimuths/yaw0/0075.png -vf "crop=iw/2:ih-28:iw/2:28" ours_f75.png
# left half  = frozen TRELLIS.2
ffmpeg -i renders_518_4azimuths/yaw0/0075.png -vf "crop=iw/2:ih-28:0:28"    frozen_f75.png
```

(The top 28 px is a text bar; the crop above removes it.)

A video of one azimuth:

```bash
ffmpeg -framerate 20 -start_number 1 -i renders_518_4azimuths/yaw0/%04d.png \
       -vf "crop=iw/2:ih-28:iw/2:28" -c:v libx264 -crf 18 -pix_fmt yuv420p ours_yaw0.mp4
```

**Frame 112 is the one we have been using for figures** — the lava is fully
developed but not yet at its most saturated. Frames 1–20 are still mostly
untextured and are misleading as a "result" still.

---

## If you want to render new cameras yourself

The texture is produced by TRELLIS.2 with the LoRA applied — there is **no baked
UV texture map to hand over**, so a new camera means re-running our renderer
rather than loading a textured mesh in Blender. Command:

```bash
python experiments/dynamesh/render_rung27_orbit.py \
  --run runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_59c51f6e \
  --sweep both --n-frames 150 --n-angles 150 \
  --turns 0 --yaw0 45 --elev 25 --res 1024 --tag my_view
```

`--turns 0` pins the camera (all change is texture), `1` is one revolution across
the sequence, `2` is two. Needs an A40 or L40S — nvdiffrast in this build has no
kernel for H100/H200.

Easier: tell me the camera and I'll render it.

---

## Numbers, if the figure quotes any

| arm | PSNR | SSIM |
|---|---|---|
| frozen TRELLIS.2 | 9.446 | 0.2670 |
| rung27 (cross-attn + self-attn LoRA) | 23.994 | — |
| rung27 + MCFM v2_D | **24.194** | **0.8347** |

150 frames, 30 epochs, rank 4, seed 42, L1 + 0.1·LPIPS, targets `qkvo+sa`.

MCFM's PSNR gain is small (+0.200 dB here, +0.074 mean over 8 objects) — its real
effect is temporal: measured as flicker (second temporal difference) it is
**−42.9% at the training view and −53.5% mean across unseen views**, and smoother
on 32 of 32 object × azimuth cells. Worth stating that way rather than as a PSNR
win, because a reviewer will check.
