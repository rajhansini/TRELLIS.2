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
