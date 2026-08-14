# Job record — TRELLIS.2 dynamesh, 12–14 Aug 2026

Auto-generated from SLURM. Regenerate with the block at the bottom.

Excludes the MCFM jobs (`m_*`, `t1_*`) which were submitted separately.


## Training runs

| job | submitted | state | elapsed | what |
|---|---|---|---|---|
| `2173382` | 2026-08-12 01:16:23 | FAILED | 00:01:19 | TRAIN  teapot(orig) @960 rung20 kv L2 |
| `2173413` | 2026-08-12 01:31:43 | COMPLETED | 01:31:29 | TRAIN  teapot(orig) @960 rung20 kv L2 |
| `2173421` | 2026-08-12 01:34:30 | COMPLETED | 01:30:21 | TRAIN  teapot_lava2 rung20 kv L2 |
| `2176410` | 2026-08-13 04:03:48 | COMPLETED | 01:34:18 | TRAIN  spot_lava rung17 kv L2  (trust map OFF) |
| `2176411` | 2026-08-13 04:03:48 | COMPLETED | 01:31:47 | TRAIN  teapot_lava2 rung17 kv L2  (trust map OFF) |
| `2176426` | 2026-08-13 04:18:17 | COMPLETED | 01:33:52 | TRAIN  spot_lava rung17 kv L1 |
| `2176427` | 2026-08-13 04:18:17 | COMPLETED | 01:34:26 | TRAIN  spot_lava rung20 kv L1 |
| `2176428` | 2026-08-13 04:18:17 | COMPLETED | 01:29:29 | TRAIN  teapot_lava2 rung17 kv L1 |
| `2176429` | 2026-08-13 04:18:17 | COMPLETED | 01:29:33 | TRAIN  teapot_lava2 rung20 kv L1 |
| `2176471` | 2026-08-13 05:04:32 | COMPLETED | 03:13:20 | TRAIN  spot_lava rung17 kv L1+LPIPS |
| `2176472` | 2026-08-13 05:04:32 | COMPLETED | 03:51:23 | TRAIN  spot_lava rung20 kv L1+LPIPS |
| `2176473` | 2026-08-13 05:04:32 | COMPLETED | 02:42:56 | TRAIN  teapot_lava2 rung17 kv L1+LPIPS |
| `2176474` | 2026-08-13 05:04:32 | COMPLETED | 02:53:02 | TRAIN  teapot_lava2 rung20 kv L1+LPIPS |
| `2177797` | 2026-08-13 22:55:42 | COMPLETED | 01:37:44 | TRAIN  spot_lava rung18 qkv L1+LPIPS |
| `2177798` | 2026-08-13 22:55:42 | COMPLETED | 01:40:07 | TRAIN  spot_lava rung19 qkvo L1+LPIPS |
| `2177799` | 2026-08-13 22:55:42 | COMPLETED | 01:33:36 | TRAIN  teapot_lava2 rung18 qkv L1+LPIPS |
| `2177800` | 2026-08-13 22:55:42 | COMPLETED | 01:36:52 | TRAIN  teapot_lava2 rung19 qkvo L1+LPIPS |
| `2177811` | 2026-08-13 23:07:31 | RUNNING | 01:30:10 | TRAIN  spot_lava rung25 kvo L1+LPIPS |
| `2177812` | 2026-08-13 23:07:31 | RUNNING | 01:28:15 | TRAIN  teapot_lava2 rung25 kvo L1+LPIPS |

## Diagnostics, smokes and renders

| job | submitted | state | elapsed | what |
|---|---|---|---|---|
| `2173255` | 2026-08-12 00:13:12 | COMPLETED | 00:04:22 | DIAG   frozen TRELLIS.2 on spot_star, fixed view |
| `2173256` | 2026-08-12 00:13:12 | COMPLETED | 00:07:15 | DIAG   frozen TRELLIS.2 on spot_star, 360 |
| `2173264` | 2026-08-12 00:16:47 | COMPLETED | 00:00:04 | DIAG   grey teapot render for the video model |
| `2173750` | 2026-08-12 03:49:42 | COMPLETED | 00:06:02 | RENDER teapot960_side |
| `2173751` | 2026-08-12 03:49:42 | COMPLETED | 00:09:16 | RENDER teapot960_360 |
| `2173752` | 2026-08-12 03:49:42 | COMPLETED | 00:07:38 | RENDER teapot960_720 |
| `2173753` | 2026-08-12 03:49:42 | COMPLETED | 00:06:06 | RENDER lava2_side |
| `2173754` | 2026-08-12 03:49:42 | COMPLETED | 00:09:23 | RENDER lava2_360 |
| `2173755` | 2026-08-12 03:49:42 | COMPLETED | 00:07:41 | RENDER lava2_720 |
| `2176293` | 2026-08-13 02:30:23 | FAILED | 00:01:40 | TEST   resolution 1024 -> OOM 43.35/44.4 GiB. 1024 closed |
| `2176443` | 2026-08-13 04:32:30 | COMPLETED | 00:16:24 | SMOKE  LPIPS calibration -> w=0.1 is 10% of loss |
| `2176488` | 2026-08-13 05:14:51 | FAILED | 00:00:01 | SMOKE  KL attention regulariser (6 attempts) |
| `2176505` | 2026-08-13 05:33:30 | FAILED | 00:01:13 | SMOKE  KL attention regulariser (6 attempts) |
| `2176515` | 2026-08-13 05:49:05 | FAILED | 00:01:14 | SMOKE  KL attention regulariser (6 attempts) |
| `2176530` | 2026-08-13 06:09:50 | FAILED | 00:01:51 | SMOKE  KL attention regulariser (6 attempts) |
| `2176540` | 2026-08-13 06:17:52 | COMPLETED | 00:11:06 | RENDER teap_r20l1_side |
| `2176541` | 2026-08-13 06:17:52 | COMPLETED | 00:09:19 | RENDER teap_r20l1_360 |
| `2176542` | 2026-08-13 06:17:52 | COMPLETED | 00:07:39 | RENDER teap_r20l1_720 |
| `2176543` | 2026-08-13 06:17:52 | COMPLETED | 00:06:38 | RENDER spot_r20l1_side |
| `2176544` | 2026-08-13 06:17:52 | COMPLETED | 00:09:49 | RENDER spot_r20l1_360 |
| `2176545` | 2026-08-13 06:17:52 | COMPLETED | 00:15:47 | RENDER spot_r20l1_720 |
| `2176546` | 2026-08-13 06:17:52 | COMPLETED | 00:14:10 | RENDER spot_r17l2_side |
| `2176547` | 2026-08-13 06:17:52 | COMPLETED | 00:17:22 | RENDER spot_r17l2_360 |
| `2176548` | 2026-08-13 06:17:52 | COMPLETED | 00:15:49 | RENDER spot_r17l2_720 |
| `2176549` | 2026-08-13 06:17:52 | COMPLETED | 00:06:36 | RENDER spot_r17l1_side |
| `2176550` | 2026-08-13 06:17:52 | COMPLETED | 00:09:54 | RENDER spot_r17l1_360 |
| `2176551` | 2026-08-13 06:17:52 | COMPLETED | 00:15:48 | RENDER spot_r17l1_720 |
| `2176552` | 2026-08-13 06:17:53 | COMPLETED | 00:06:02 | RENDER teap_r17l1_side |
| `2176553` | 2026-08-13 06:17:53 | COMPLETED | 00:09:21 | RENDER teap_r17l1_360 |
| `2176554` | 2026-08-13 06:17:53 | COMPLETED | 00:07:46 | RENDER teap_r17l1_720 |
| `2176555` | 2026-08-13 06:17:53 | COMPLETED | 00:05:55 | RENDER teap_r17l2_side |
| `2176556` | 2026-08-13 06:17:53 | COMPLETED | 00:14:11 | RENDER teap_r17l2_360 |
| `2176557` | 2026-08-13 06:17:53 | COMPLETED | 00:12:24 | RENDER teap_r17l2_720 |
| `2176563` | 2026-08-13 06:27:26 | FAILED | 00:01:31 | SMOKE  KL attention regulariser (6 attempts) |
| `2177337` | 2026-08-13 14:41:26 | COMPLETED | 00:06:50 | RENDER spot_r17lp_side |
| `2177338` | 2026-08-13 14:41:26 | COMPLETED | 00:09:55 | RENDER spot_r17lp_360 |
| `2177339` | 2026-08-13 14:41:26 | COMPLETED | 00:08:22 | RENDER spot_r17lp_720 |
| `2177340` | 2026-08-13 14:41:26 | COMPLETED | 00:06:45 | RENDER spot_r20lp_side |
| `2177341` | 2026-08-13 14:41:26 | COMPLETED | 00:09:49 | RENDER spot_r20lp_360 |
| `2177342` | 2026-08-13 14:41:26 | COMPLETED | 00:08:19 | RENDER spot_r20lp_720 |
| `2177343` | 2026-08-13 14:41:26 | COMPLETED | 00:06:05 | RENDER teap_r17lp_side |
| `2177344` | 2026-08-13 14:41:26 | COMPLETED | 00:09:22 | RENDER teap_r17lp_360 |
| `2177345` | 2026-08-13 14:41:26 | COMPLETED | 00:07:41 | RENDER teap_r17lp_720 |
| `2177346` | 2026-08-13 14:41:26 | COMPLETED | 00:05:58 | RENDER teap_r20lp_side |
| `2177347` | 2026-08-13 14:41:26 | COMPLETED | 00:09:21 | RENDER teap_r20lp_360 |
| `2177348` | 2026-08-13 14:41:26 | COMPLETED | 00:07:33 | RENDER teap_r20lp_720 |
| `2177541` | 2026-08-13 16:42:54 | COMPLETED | 00:03:06 | SMOKE  KL attention regulariser (6 attempts) |
| `2177544` | 2026-08-13 16:46:32 | COMPLETED | 00:02:45 | SMOKE  KL attention regulariser (6 attempts) |
| `2177550` | 2026-08-13 16:50:44 | COMPLETED | 00:03:07 | SMOKE  KL attention regulariser (6 attempts) |
| `2177577` | 2026-08-13 17:05:25 | COMPLETED | 00:08:24 | CALIB  KL magnitude vs |B| -> KL ~ |B|^2, beta~3 for 10% |
| `2177631` | 2026-08-13 17:50:35 | COMPLETED | 00:05:32 | DIAG   attention entropy seen vs unseen voxels |
| `2177639` | 2026-08-13 17:57:57 | COMPLETED | 00:05:32 | DIAG   attention entropy seen vs unseen voxels |
| `2177792` | 2026-08-13 22:49:22 | COMPLETED | 00:03:06 | SMOKE  rung18 qkv gates |
| `2177793` | 2026-08-13 22:49:22 | COMPLETED | 00:03:08 | SMOKE  rung19 qkvo gates |
| `2177805` | 2026-08-13 23:00:54 | COMPLETED | 00:06:01 | RENDER kvo_side |
| `2177843` | 2026-08-13 23:28:40 | COMPLETED | 00:00:11 | DIAG   SAM2 vs threshold segmentation |

**19 training jobs, 56 diagnostics/renders.**

```bash
# live status, results and videos:
bash experiments/dynamesh/JOBS.sh
```
