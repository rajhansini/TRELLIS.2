# KL divergence — job ledger

Every job submitted for `rung30_kl_both.py` (LoRA on cross **and** self-attention,
`qkvo+sa`, with KL on **both** attention maps, independently weighted by `β_c` and `β_s`).

Pipeline per object: **probe** (6 ep, `--kl-probe`) → **16-cell grid** → **β=0 control**.
β is derived from the object's own probe: `β = f/(1−f) · loss/KL`, f ∈ {15,25,50,75}%.

---

## monster_rainbow — submitted 2026-08-20

| stage | job | config | state |
|---|---|---|---|
| probe | **2190858** | 6 ep, 150 fr, `--kl-probe --w-kl 0 --w-kl-self 0` | PENDING |
| grid c15_s15 … c75_s75 | 16 jobs | auto-submitted from probe ratios | awaiting probe |
| control c0_s0 | 1 job | `--w-kl 0 --w-kl-self 0`, 30 ep | awaiting probe |

mesh `data/monster_rainbow/mesh/monster_rainbow_render_frame.obj`
targets `out/gt_targets_monster_rainbow/frames` — 150 gt_*.png, mask 160,641 px
video `data/monster_rainbow/frames_from_video` — 150 frames
rung27 baseline (mcfm v2_D): **24.47 dB**

## skull_lava — submitted 2026-08-20

| stage | job | config | state |
|---|---|---|---|
| probe | **2190859** | 6 ep, 150 fr, `--kl-probe --w-kl 0 --w-kl-self 0` | PENDING |
| grid c15_s15 … c75_s75 | 16 jobs | auto-submitted from probe ratios | awaiting probe |
| control c0_s0 | 1 job | `--w-kl 0 --w-kl-self 0`, 30 ep | awaiting probe |

mesh `data/skull_lava/mesh/skull_lava_render_frame.obj`
targets `out/gt_targets_skull_lava/frames` — 150 gt_*.png, mask 264,786 px
video `data/skull_lava/frames_from_video` — 150 frames
rung27 baseline (mcfm v2_D): **25.68 dB**

Neither object carries a `_guan` suffix, so both were added as an explicit case in
`jobs/submit_r30_grid.sh`; the default branch would have pointed at
`gt_targets_<obj>_guan/` and `<obj>_render_frame_guan.obj`, neither of which exists.

---

## Completed grids

| object | grid | control | probe KLc / KLs | control dB | best cell | worst cell | beat control |
|---|---|---|---|---|---|---|---|
| horse_metal | 16 | ✓ | — | 26.017 | 25.519 | 24.645 | **0/16** |
| penguin_circuits | 16 | ✓ | — | 24.570 | 23.917 | 23.160 | **0/16** |
| whale_spots | 16 | ✓ | — | 24.636 | 23.783 | 22.779 | **0/16** |
| teapot_porcelain | 16 | ✓ | — | 19.508 | 18.700 | 17.706 | **0/16** |
| spot_lava | 16 | ✓ | — | 24.194 | 23.424 | 22.996 | **0/16** |
| teapot_lava2 | 16 | ✓ | — | 22.688 | 22.180 | 21.214 | **0/16** |
| teapot_ceramic_crack_correct | 16 | 2183384 | 5.426 / 1.665 | 25.398 | 24.718 | 23.897 | **0/16** |

**112 runs completed, 0 beat their control.** Monotone in both β on every object.

Artifact (ceramic_correct, 16 turntables): https://claude.ai/code/artifact/98693d29-e196-48c8-8db5-d009b7880e86

---

## Not run

| script | KL on | status |
|---|---|---|
| `rung29_kl_cross_attention.py` | cross only, rung27 base | 24 run dirs, **0 finished** — 30 FAILED, 24 CANCELLED, 6 TIMEOUT. Scripts at `jobs/r29arm_<obj>_p{15,25,50,75}.sbatch` |
| `rung28_kl_selfattn.py` | self only, rung27 base | script written, **never launched**, no sbatch files |

These are the isolation arms. rung30 shows both terms together hurt; nothing yet attributes
the damage to the cross term, the self term, or their interaction.

---

## How to check state

```bash
squeue -u rajhansini -o "%.10i %.30j %.9T" | grep -E "r30probe|r30g_"        | tee -a ~/logs/kl_state.log
grep -iE "r30probe|r30g_" /net/projects/ranalab/rajhansini/JOBLOG.tsv | cut -f1,3 | tee -a ~/logs/kl_state.log
```
