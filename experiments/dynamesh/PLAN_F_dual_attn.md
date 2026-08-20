# F · Temporal vs spatial attention — rung32

Answers the reviewer question: *you put LoRA on spatial cross-attention; why not on
temporal?* Today temporal mixing is MCFM — parameter-free, applied to the tokens
before the flow sees them. This makes it **learned** and pits the two against each other.

## Design

Two branches, same frozen cross-attention weights, two separate LoRA sets:

    spatial branch   ctx = C_t                1029 tokens of frame t     (current method)
    temporal branch  ctx = C_{t-1..t+1}[i]    W items at token position i  (time only)

    out = spatial + g · temporal            g = per-block scalar, zero-init

- Frozen weights are **shared, not copied** — "duplicating the matrices" costs 0 params.
  Only the LoRA sets and g differ. Run `cross_attn` twice with different ctx.
- LoRA here is hook-based (`blk.cross_attn.to_q/to_kv/to_out`), so the branch is
  selected by a flag the hook reads. No module surgery.
- `g=0` at init ⇒ step 0 is bit-identical to frozen. Non-negotiable; every baseline
  assumes it.

## Arms — 3 × 6 objects = 18 runs

| arm | LoRA on | mcfm | purpose |
|-----|---------|------|---------|
| F1 | spatial only | off | **correctness gate** — must reproduce rung27 exactly |
| F2 | temporal only | off | learned temporal, no spatial adaptation |
| F3 | both | off | the full claim |

MCFM is **off** in all three. F2/F3 already mix over time; leaving MCFM on would
double-count it and make the comparison meaningless.

## What it is compared against (already measured)

    rung27                      23.994    no temporal mixing at all
    rung27 + MCFM v2_D          24.194    parameter-free temporal   <-- the bar
    F2 / F3                        ?      learned temporal

The paper claim lands either way:
- F3 > 24.194 → learned temporal attention beats parameter-free blending.
- F3 ≈ 24.194 → MCFM gets the same gain for zero parameters. **Stronger result.**

## Build order

1. Copy `rung27_selfattn_lora.py` → `rung32_dual_attn_lora.py`. Bump `_RUNG` **and**
   `_CFG['rung']` together — they are asserted equal (this killed 24 runs on rung29).
2. Add to `_CFG`: `dual_attn`, `branch`, `temporal_window`, `gate_init`. Anything not
   in `_CFG` does not reach the config hash, and arms collide into one directory
   (this killed all 24 rung30 arms the first time).
3. `--branch {spatial,temporal,both}` and `--temporal-window {3,5}`.
4. Second LoRA registry + per-block zero-init gate.
5. Temporal ctx builder — reuse `mcfm_blend.py`'s `_OFFSETS`, edge frames clamp.

## Gates before any grid is submitted

- **G1** `--branch spatial` reproduces rung27's PSNR to <0.01 dB on spot_lava. If not,
  the refactor changed something and every F number is void.
- **G2** gate=0 + LoRA zero ⇒ output bit-identical to frozen.
- **G3** smoke run: 2 epochs, 8 frames, confirm step time and peak VRAM. Two
  cross-attn passes per block ≈ 1.5–1.8× compute; KL already needed ~16 GiB, so
  L40S, not rtx2080ti.

## Risks

- Cost is the main one — 30 blocks × 2 cross-attn passes. If VRAM blows, fall back to
  temporal branch on `--blocks late` only.
- Gradient checkpointing replays the forward in backward; the branch flag must read
  identically in both passes or you get a CheckpointError tensor-count mismatch
  (this happened on rung30).
- 18 runs, one sbatch each, one per object per arm. Never loop arms inside a job.

## Open question

Temporal branch attends **per token position over W frames** (time only, mirrors MCFM
v2). The alternative is joint over W×1029 tokens (mirrors v3). Per-position is the
cheap one and the direct contrast with v2_D, which is what is already measured on all
6 objects. Confirm before step 5.
