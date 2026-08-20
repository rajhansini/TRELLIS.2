# KL divergence on both attention maps — rung30

`rung30_kl_both.py` — rung27 LoRA (`qkvo+sa`, cross **and** self-attention, 2,334,720 params)
plus a KL penalty holding **both** attention maps near frozen TRELLIS.2, independently weighted.

```
L = L1 + 0.1·LPIPS + β_c · KL_cross + β_s · KL_self

KL_cross   softmax(q kᵀ/√d) over the 1,029 image tokens    "where do I look in the photo"
KL_self    softmax(q kᵀ/√d) over the 1,999 voxel tokens    "who do I listen to in 3D"
```

**β is measured, never chosen.** Each object is probed for 6 epochs at β=0 to get its own
`loss/KLc` and `loss/KLs`, then `β = f/(1−f) · ratio` for f ∈ {15, 25, 50, 75}%. An earlier
attempt used β=1.0 and the term was ~0.1% of the gradient — that run tested nothing.

**Caveat for the paper:** KL grows as ‖B‖² while the data loss falls, so `loss/KL` is still
dropping ~30%/epoch when the 6-epoch probe stops. Every cell shares the convention so the
ordering holds, but the percentage labels are not exact shares of the converged loss.

---

## Pipeline per object

| # | step | command | output |
|---|---|---|---|
| 1 | probe | `sbatch jobs/r30probe_<obj>.sbatch` | `loss/KLc`, `loss/KLs` in the log |
| 2 | grid | `bash jobs/submit_r30_grid.sh <obj>` | 16 jobs, `r30grid_<obj>_c{15,25,50,75}_s{...}.sbatch` |
| 3 | control | `--w-kl 0 --w-kl-self 0` | the β=0 reference the grid is scored against |

`submit_r30_grid.sh` derives mesh / target-dir / frame-count per object. Objects **without** the
`_guan` suffix need an explicit case or the default branch points at paths that do not exist.

---

## Completed

| object | cells | frames | probe loss/KLc | probe loss/KLs | control β=0 | best cell | worst cell | verdict |
|---|---|---|---|---|---|---|---|---|
| horse_metal | 16 | 121 | — | — | 26.017 | 25.519 | 24.645 | 0/16 beat control |
| penguin_circuits | 16 | 121 | — | — | 24.570 | 23.917 | 23.160 | 0/16 |
| whale_spots | 16 | 121 | — | — | 24.636 | 23.783 | 22.779 | 0/16 |
| teapot_porcelain | 16 | 121 | — | — | 19.508 | 18.700 | 17.706 | 0/16 |
| spot_lava | 16 | 150 | — | — | 24.194 | 23.424 | 22.996 | 0/16 |
| teapot_lava2 | 16 | 150 | — | — | 22.688 | 22.180 | 21.214 | 0/16 |
| teapot_ceramic_crack_correct | 16 | 150 | 5.426 | 1.665 | **25.398** | 24.718 (c15_s15) | 23.897 (c75_s75) | **0/16** |

**112 runs, 0 beat their control.** Monotone in both weights on every object — no interior
optimum anywhere, so this is not a tuning failure.

Artifact: https://claude.ai/code/artifact/98693d29-e196-48c8-8db5-d009b7880e86 (ceramic_correct, 16 turntables)

---

## Submitted — awaiting probe

| object | frames | mesh | targets | probe job | grid |
|---|---|---|---|---|---|
| monster_rainbow | 150 | `monster_rainbow_render_frame.obj` | `gt_targets_monster_rainbow/frames` (150) | **2190858** | pending probe |
| skull_lava | 150 | `skull_lava_render_frame.obj` | `gt_targets_skull_lava/frames` (150) | **2190859** | pending probe |

Verified before submission — 150 GT frames each, 150 video frames each, masks
160,641 px (monster_rainbow) and 264,786 px (skull_lava). Neither carries a `_guan` suffix,
so both were added as an explicit case in `submit_r30_grid.sh`.

Their rung27 baselines: monster_rainbow 24.47, skull_lava 25.68 (mcfm v2_D arm).

---

## Related but different

`kl_attn_lora.py` — the **earlier** experiment: LoRA on cross-attention only (`--targets kv`,
rung17 base) with KL on cross-attention only. 8 arms, 2 objects, also monotone.
Artifact: https://claude.ai/code/artifact/2c4673df-6666-4eb7-8d72-17ffb7ec2642

`rung29_kl_cross_attention.py` — cross-only KL on the rung27 base. 24 run dirs, **none finished**
(30 FAILED, 24 CANCELLED, 6 TIMEOUT). Job scripts exist as `r29arm_<obj>_p{15,25,50,75}.sbatch`.

`rung28_kl_selfattn.py` — self-only KL on the rung27 base. Script written, **never launched**.

Those two are the missing isolation arms: rung30 shows both terms together hurt, but nothing
yet attributes the damage to the cross term, the self term, or their interaction.
