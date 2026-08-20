# CVPR ablations — what to run

Base config for every arm unless stated: `rung27_selfattn_lora.py --targets qkvo+sa
--recon l1 --lpips --w-lpips 0.1 --rank 4 --seed 42 --epochs 30 --blocks all`.
Objects (6): spot_lava, teapot_lava2, horse_metal, penguin_circuits, whale_spots,
teapot_porcelain.  One sbatch per object per arm. A40/L40S only.

Status legend:  HAVE = final_eval.json exists   NEED = to run   CODE = script work first

---

## A · Component ladder (main paper table)

| # | stage | flags | have |
|---|-------|-------|------|
| 1 | frozen | no training, eval only | 6/6 |
| 2 | + cross-attn | rung19 `--targets qkvo` | 6/6 |
| 3 | + aggregation | **UNDEFINED — see Q1** | ? |
| 4 | + self-attn | rung27 `--targets qkvo+sa` | 6/6 |
| 5 | + temporal (MCFM) | `--mcfm v2_D` | 6/6 |

**NEED: 0 runs** except stage 3 once Q1 is answered. The ladder is already measured.
Cumulative: each row keeps everything above it.

## B · LPIPS on/off

Drop `--lpips` from stage 5. Isolates the perceptual term.

- HAVE 1 (horse_metal, rung27 no-mcfm) — not the right cell
- **NEED 6** — rung27 `--mcfm v2_D --recon l1` (no `--lpips`), all 6 objects

## C · LoRA rank (supplementary)

`--rank {1,2,8,16,32}` at stage 5. Existing sweep is on rung22/teapot, obsolete.

- **NEED 10** — 5 ranks × 2 objects (spot_lava, horse_metal). Supplementary only;
  6 objects would be 30 and buys little.

## D · Temporal window size  ← the reviewer-facing one

Window = how many frames MCFM blends. Too wide = over-averaged, texture washes out.

| W | frames | mode | have | need |
|---|--------|------|------|------|
| 1 | [t] — no temporal | rung27, no `--mcfm` | 6/6 | 0 |
| 2 | [t, t+1] | `--mcfm v2_C` | 2/6 | **4** |
| 3 | [t-1, t, t+1] | `--mcfm v2_D` | 6/6 | 0 |
| 5 | [t-2 … t+2] | `--mcfm v2_E` | 0/6 | **6** |

**CODE:** `mcfm_blend.py` only defines `_OFFSETS = {'C': (0,1), 'D': (-1,0,1)}`.
Add `'E': (-2,-1,0,1,2)` and extend `MODES`. ~4 lines. W=5 does not exist yet.

**NEED 10 runs + code.** This is the curve that shows the optimum is interior
(W=3), which is the claim: some temporal sharing helps, too much destroys texture.

## E · Blend variant, per-position vs joint

`v2` = 1029 independent softmaxes over the window. `v3` = one softmax over time AND
space. Same window (D).

- HAVE 2 (v3_D on f0075, spot_lava) — **NEED 4**

## F · Temporal vs spatial attention — NEW EXPERIMENT

The reviewer question: *you adapt spatial cross-attention; why not temporal?*

Duplicate the cross-attention matrices into two branches — one blends tokens
**spatially** (normal), one blends **temporally** — and put LoRA on each.

| arm | LoRA on | need |
|-----|---------|------|
| F1 | spatial branch only (= current method) | 6 |
| F2 | temporal branch only | 6 |
| F3 | both branches | 6 |

**CODE:** new `rung32_dual_attn_lora.py`, copied from rung27. Largest item here —
duplicating the branch, routing the token blend, and a `--targets` extension for
`temporal`/`spatial`/`both`. **NEED 18 runs + a new script.**

## G · Frame count

`--n-frames {30, 60, 90, all}` at stage 5 — does the method need the full sequence?

- **NEED 6** — 3 short settings × 2 objects (spot_lava, whale_spots)

## H · Limitation — stochastic vs structured texture

No new training. Group the 6 existing objects by texture class and report PSNR by
class: stochastic (lava, spots, circuits) vs structured (metal, porcelain). The
claim is that structured textures degrade because token blending cannot preserve
alignment of a regular pattern.

- **NEED 0 runs** — analysis + one figure from renders already on disk

---

## Totals

| | runs | code |
|---|---|---|
| B lpips | 6 | — |
| C rank | 10 | — |
| D window | 10 | mcfm_blend.py, ~4 lines |
| E variant | 4 | — |
| F temporal/spatial | 18 | new rung32 script |
| G frames | 6 | — |
| **total** | **54** | 2 items |

Plus one 720° render per run. Every submission → JOBLOG.tsv via joblog.sh.

---

## Open questions — answer before anything is submitted

**Q1 · What is "aggregation" in the ladder?** rung17 is backprojection, rung19 is
the intersection mask. Which one is the paper's aggregation stage, and does it sit
between cross-attn and self-attn?

**Q2 · teapot_lava2 does not appear in the run census.** 50 runs are tagged
spot_lava, but teapot_lava2 shows up only under ambiguous mesh paths. Confirm it is
in the 6, and which runs are its.

**Q3 · Seeds.** Everything is seed 42 only. The MCFM step in the main table is
+0.200 dB — small enough that a reviewer will ask if it is seed noise. Stage 4 and
stage 5 need ≥3 seeds each (12 extra runs) or the claim is not defensible. The
+13.224 and +1.324 steps are far too large to need this.
