# temporal_attention — the three-mode conditioning comparison

Question: with the model FROZEN and no adapter anywhere, does the choice of
temporal blend change the conditioning enough to change the render?

Three modes, one window `[t-1, t, t+1]`, everything else pinned:

| mode | what it reads | legacy code |
|---|---|---|
| `temporal_only` | token j across the window; no spatial term | v2 / phase d v2 |
| `spatial_then_temporal` | spatial self-attn on frame t, then temporal | v2b / phase d v2b |
| `joint_spatiotemporal` | one softmax over all W x N tokens | v3 / phase d v3 |

`temporal_then_spatial` is deliberately EXCLUDED: the frozen cross-attention
already performs the spatial pass, so an explicit one duplicates it.

## Why both repos

|  | TRELLIS 1 | TRELLIS.2 |
|---|---|---|
| geometry | generated from a reference frame | fixed input mesh |
| DINO | v2, 518px, 1374 tokens | v3, 512px, 1029 tokens |
| backbone | TRELLIS-image-large | TRELLIS.2-4B |
| adapter | none | none (`--frozen-only`) |

The blend operators are byte-identical across repos (verified 0.00e+00). Everything
around them differs, so absolute numbers do NOT transfer -- only the RANKING of the
three modes within a repo is comparable. That is the whole reason for running both:
if the ranking agrees across two different backbones and two different DINOs, it is
a property of the blend and not of one pipeline.

## Runs

<!-- appended by log_ta.sh: date | jobid | repo | object | mode | outdir | state -->
08-20 00:09 | 2190706 | TRELLIS.2 | spot_lava | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_spot_lava_temporal_only_w3/frames
08-20 00:09 | 2190707 | TRELLIS1 | spot_lava | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_spot_lava_v2_phase_d_v2
08-20 00:09 | 2190708 | TRELLIS.2 | spot_lava | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_spot_lava_spatial_then_temporal_w3/frames
08-20 00:09 | 2190709 | TRELLIS1 | spot_lava | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_spot_lava_v2b_phase_d_v2b
08-20 00:09 | 2190710 | TRELLIS.2 | spot_lava | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_spot_lava_joint_spatiotemporal_w3/frames
08-20 00:09 | 2190711 | TRELLIS1 | spot_lava | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_spot_lava_v3_phase_d_v3
08-20 00:09 | 2190712 | TRELLIS.2 | hand_rorschach | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_hand_rorschach_temporal_only_w3/frames
08-20 00:09 | 2190713 | TRELLIS1 | hand_rorschach | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_hand_rorschach_v2_phase_d_v2
08-20 00:09 | 2190714 | TRELLIS.2 | hand_rorschach | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_hand_rorschach_spatial_then_temporal_w3/frames
08-20 00:09 | 2190715 | TRELLIS1 | hand_rorschach | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_hand_rorschach_v2b_phase_d_v2b
08-20 00:09 | 2190716 | TRELLIS.2 | hand_rorschach | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_hand_rorschach_joint_spatiotemporal_w3/frames
08-20 00:09 | 2190717 | TRELLIS1 | hand_rorschach | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_hand_rorschach_v3_phase_d_v3
08-20 00:09 | 2190718 | TRELLIS.2 | skull_lava | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_skull_lava_temporal_only_w3/frames
08-20 00:09 | 2190719 | TRELLIS1 | skull_lava | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_skull_lava_v2_phase_d_v2
08-20 00:09 | 2190720 | TRELLIS.2 | skull_lava | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_skull_lava_spatial_then_temporal_w3/frames
08-20 00:09 | 2190721 | TRELLIS1 | skull_lava | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_skull_lava_v2b_phase_d_v2b
08-20 00:09 | 2190722 | TRELLIS.2 | skull_lava | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_skull_lava_joint_spatiotemporal_w3/frames
08-20 00:09 | 2190723 | TRELLIS1 | skull_lava | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_skull_lava_v3_phase_d_v3
08-20 00:09 | 2190724 | TRELLIS.2 | pumpkin_rot | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_pumpkin_rot_temporal_only_w3/frames
08-20 00:09 | 2190725 | TRELLIS1 | pumpkin_rot | temporal_only | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_pumpkin_rot_v2_phase_d_v2
08-20 00:09 | 2190726 | TRELLIS.2 | pumpkin_rot | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_pumpkin_rot_spatial_then_temporal_w3/frames
08-20 00:09 | 2190727 | TRELLIS1 | pumpkin_rot | spatial_then_temporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_pumpkin_rot_v2b_phase_d_v2b
08-20 00:09 | 2190728 | TRELLIS.2 | pumpkin_rot | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_pumpkin_rot_joint_spatiotemporal_w3/frames
08-20 00:09 | 2190729 | TRELLIS1 | pumpkin_rot | joint_spatiotemporal | /net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement/results_mcfm_ta1_pumpkin_rot_v3_phase_d_v3

## rung32 — wide context window (submitted 08-20 03:38)

Sibling of the MCFM arm above, testing the same hypothesis by a different route.
MCFM *blends* the cached DINOv3 tokens of neighbouring frames into one conditioning
tensor before cross-attention. rung32 does not blend: it **concatenates** frames
`[f-1, f, f+1]` and lets cross-attention see all three (1029 -> 3087 tokens), so the
model chooses its own weighting instead of being handed a fixed one. Everything else
is rung27 exactly — `qkvo+sa`, rank 4, alpha 4, seed 42, 30 epochs, 960 px, l1+lpips.
`--context-window` and `--mcfm` are mutually exclusive.

Note the run directory says **rung27**, not rung32: `_RUNG` keys off `_HAS_SA`, and
the context window rides in the `_cw3` label tag plus the config hash. Look for
`runs/rung27_l1_lp_cw3_all_qkvo+sa_r4_s42_<hash>/`.

| date | jobid | object | mesh | targets | frames | run dir |
|---|---|---|---|---|---|---|
| 08-20 03:38 | 2191090 | spot_lava | spot_render_frame.obj | gt_targets_spot_lava | 150 | runs/rung27_l1_lp_cw3_all_qkvo+sa_r4_s42_* |
| 08-20 03:38 | 2191091 | skull_lava | skull_lava_render_frame.obj | gt_targets_skull_lava | 150 | runs/rung27_l1_lp_cw3_all_qkvo+sa_r4_s42_* |
| 08-20 03:38 | 2191092 | hand_rorschach | hand_rorschach_render_frame.obj | gt_targets_hand_rorschach | 150 | runs/rung27_l1_lp_cw3_all_qkvo+sa_r4_s42_* |
| 08-20 03:38 | 2191093 | pumpkin_rot | pumpkin_rot_render_frame_guan.obj | gt_targets_pumpkin_rot_guan | 121 | runs/rung27_l1_lp_cw3_all_qkvo+sa_r4_s42_* |

PASS per job: 30 rows in `epoch_metrics.csv` + `final_eval.json`.

**Baselines already trained**, for the 4-way panel these feed:

| object | rung27 | +MCFM v2_D | rung31 (tw3b) |
|---|---|---|---|
| spot_lava | 23.99 | 24.19 | 24.17 |
| skull_lava | 25.35 | 25.68 | 25.63 |
| hand_rorschach | 24.83 | 25.06 | 24.96 |
| pumpkin_rot | 33.72 | 33.82 | 33.79 |

PSNR, dB. rung31's `tw3b` tag is `temporal_window=3, branch=both, gate_init=0.0` —
a gated temporal branch, *not* the warm-start the filename `rung31_warmstart_lora.py`
suggests. Worth knowing before reading these numbers as an init-from ablation.

### Resume logic

Every partition here caps at 4 h and `--requeue` does **not** cover TIMEOUT, only
node failure and preemption. Plain rung27 already takes ~1h54m at 150 frames and
rung32 triples the cross-attention token count, so hitting the wall mid-run is the
expected case. `jobs/supervise_rung32.py` (running under nohup, log
`out/rung32_supervisor.log`) polls every 3 min and resubmits any object that leaves
the queue without a `final_eval.json`, up to 6 retries over 30 h. The trainer
checkpoints every epoch and has GATE-resume, so a resubmit continues from the last
completed epoch rather than restarting.

Constraint is `a40|L40S`; `a30` was removed because the nvdiffrast build in the
trellis2 env only carries sm86/sm89 kernels and dies on sm80 at the first rasterize.
### Baselines to beat (temporal_only_w3, already trained)

| object | rung27 (no MCFM) | + temporal_only_w3 | delta |
|---|---|---|---|
| spot_lava | — | — | — |
| skull_lava | 25.347 | 25.685 | +0.338 |
| hand_rorschach | 24.831 | 25.057 | +0.226 |
| pumpkin_rot | — | — | — |

### Resume note

Training resumes from the last per-epoch checkpoint, optimizer state included, and
GATE-resume refuses a checkpoint written against a different mesh or target set. So
a 4h wall-clock kill costs ~1 epoch (~6 min), not the run. Renders and panels are
`--requeue` and skip work already on disk. The watchdog resubmits infrastructure
failures but deliberately will NOT retry a GATE failure -- those need a fix, and it
logs them as NEEDS-ATTENTION instead.

Check in the morning with:

    cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
    sacct -j 2191095,2191096,2191097,2191098,2191099,2191100,2191101 \
      -X --format=JobID,JobName%22,State,Elapsed


## rung27 x 3 MCFM modes — submitted 2026-08-20

Same rung27 adapter on every arm (`qkvo+sa`, rank 4, seed 42, 30 epochs, L1+0.1 LPIPS).
The ONLY difference is which blend produced the conditioning. `temporal_only_w3` already
existed for all four objects and `joint_spatiotemporal_w3` for spot_lava, so only the
seven missing arms were submitted.

| job | object | mode | code |
|---|---|---|---|
| 2191095 | spot_lava | spatial_then_temporal_w3 | st_D |
| 2191096 | skull_lava | spatial_then_temporal_w3 | st_D |
| 2191097 | skull_lava | joint_spatiotemporal_w3 | v3_D |
| 2191098 | hand_rorschach | spatial_then_temporal_w3 | st_D |
| 2191099 | hand_rorschach | joint_spatiotemporal_w3 | v3_D |
| 2191100 | pumpkin_rot | spatial_then_temporal_w3 | st_D |
| 2191101 | pumpkin_rot | joint_spatiotemporal_w3 | v3_D |

Each job reads its mesh, targets and frame count from that object's OWN existing rung27
config rather than assuming the hero layout. pumpkin_rot is on the `_guan` pipeline at
121 frames -- hardcoding hero paths would have trained it against the wrong targets and
never raised.

| object | rung27 (no MCFM) | + temporal_only_w3 | delta |
|---|---|---|---|
| spot_lava | 23.994 | 24.194 | +0.200 |
| skull_lava | 25.347 | 25.685 | +0.337 |
| hand_rorschach | 24.831 | 25.057 | +0.226 |
| pumpkin_rot | 33.723 | 33.816 | +0.092 |

Read the delta column as the bar the other two modes have to clear. It is small
(+0.09 to +0.34 dB) because PSNR scores each frame against its own target and cannot
see a temporal prior -- the mode difference, if there is one, should show as flicker
and on the unseen views, not here.

## Resume note

Training resumes from the last per-epoch checkpoint, optimizer state included, and
GATE-resume refuses a checkpoint written against a different mesh or target set. A 4h
wall-clock kill therefore costs ~1 epoch (~6 min), not the run. Renders and panels are
`--requeue` and skip work already on disk. The watchdog resubmits infrastructure
failures but deliberately will NOT retry a GATE failure -- those need a fix, and it
records them as NEEDS-ATTENTION instead.

Morning check:

    cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
    sacct -j 2191095,2191096,2191097,2191098,2191099,2191100,2191101 \
      -X --format=JobID,JobName%22,State,Elapsed
    ./report_ta.sh          # the TRELLIS 1 / TRELLIS.2 three-mode sweep
