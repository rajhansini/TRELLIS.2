"""
mcfm_blend.py — MCFM temporal token blending, as a standalone importable module.

NEW FILE. Nothing existing is modified. rung20_conf_lora.py, kl_attn_lora.py and
TRELLIS.2's own source are untouched; a caller opts in by importing blend_conds()
and passing --mcfm on its own command line.

WHAT IT DOES
  MCFM replaces each frame's image-conditioning tokens with a blend over a short
  temporal window, BEFORE the flow model sees them. Neighbouring frames then
  share conditioning, so their textures cannot drift apart independently.
  Parameter-free: nothing here is trained, and with mode=None it is a no-op.

  v2  per-position : token i of frame t attends over token i of the window only.
                     1029 independent softmaxes over W items.
  v3  joint        : frame t's N tokens query a pooled bank of W*N tokens —
                     one softmax over time AND space together.
  C   window [t, t+1]        D   window [t-1, t, t+1]

VALIDATED IN TRELLIS 1 FIRST
  The same four modes were run on spot_lava and teapot_lava2 through TRELLIS 1
  (experiments/enhancement/step06b_asset.py, 8 runs, 150 frames each) and the
  renders confirmed the blend produces coherent moving texture. The tokens do NOT
  transfer — TRELLIS 1 is DINOv2 (1374, 1024), TRELLIS.2 is DINOv3 (1029, 1024) —
  but MCFM has no learned parameters, so only the operation carries over.

N IS NEVER HARDCODED
  TRELLIS 1's step4_mcfm/mcfm.py pins N_TOKENS = 1374, which is a DINOv2-at-518
  number. Here N and D are read from the tensor, so 512 (1029 tokens) and 1024
  (4101 tokens) both work with no change.

MEMORY
  v2 stacks (N, W, D); v3 forms an (N, W*N) attention matrix — at resolution 1024
  with window D that is 4101 x 12303, about 202 MB in fp32. Both are negligible
  against the flow model's own footprint (measured 10.6 GiB at 512, 39.2 GiB at
  1024 for frozen inference).
"""

from typing import Dict, Optional

import torch

MODES = ('v2_C', 'v2_D', 'v3_C', 'v3_D')
_OFFSETS = {'C': (0, 1), 'D': (-1, 0, 1)}


def parse_mode(mode: str):
    """'v2_D' -> ('v2', (-1, 0, 1), 1). Raises on anything unrecognised."""
    if mode not in MODES:
        raise ValueError(f'--mcfm must be one of {MODES}, got {mode!r}')
    variant, window = mode.split('_')
    offsets = _OFFSETS[window]
    return variant, offsets, offsets.index(0)


def blend_window(window_toks, t_pos: int, variant: str) -> torch.Tensor:
    """window_toks: list of (N, D) tensors, current frame at index t_pos."""
    cur = window_toks[t_pos]                                  # (N, D)
    D = cur.shape[-1]
    scale = D ** -0.5
    stack = torch.stack(window_toks, dim=1)                   # (N, W, D)
    if variant == 'v2':
        q = cur.unsqueeze(1)                                  # (N, 1, D)
        attn = torch.softmax(torch.bmm(q, stack.transpose(1, 2)) * scale, dim=-1)
        return torch.bmm(attn, stack).squeeze(1)              # (N, D)
    pool = stack.reshape(-1, D)                               # (W*N, D)
    attn = torch.softmax(torch.mm(cur, pool.T) * scale, dim=-1)
    return torch.mm(attn, pool)                               # (N, D)


@torch.no_grad()
def blend_conds(conds: Dict[int, torch.Tensor],
                mode: Optional[str],
                verbose: bool = True) -> Dict[int, torch.Tensor]:
    """Blend a {frame_index -> cond} dict in place of the vanilla tokens.

    conds values may be (N, D) or (1, N, D); the batch dim is preserved so the
    result is a drop-in replacement for whatever the caller already had.

    Frames outside the available range are CLAMPED to the nearest existing frame
    (frame 1's t-1 is frame 1), so the first and last frames blend over a
    truncated window rather than wrapping or dropping out.

    mode=None returns conds unchanged, so a caller can leave the call site in
    place and select vanilla behaviour from the command line.
    """
    if mode is None:
        if verbose:
            print('[MCFM] disabled — vanilla tokens unchanged', flush=True)
        return conds

    variant, offsets, t_pos = parse_mode(mode)
    keys = sorted(conds.keys())
    lo, hi = keys[0], keys[-1]
    present = set(keys)

    def _at(i):
        j = min(max(i, lo), hi)
        while j not in present and lo < j < hi:      # tolerate gaps in the index
            j += 1 if i > t_pos else -1
        return conds[j if j in present else keys[0]]

    out = {}
    for fi in keys:
        win = []
        for o in offsets:
            t = _at(fi + o)
            win.append(t[0] if t.dim() == 3 else t)
        blended = blend_window(win, t_pos, variant)
        ref = conds[fi]
        out[fi] = blended.unsqueeze(0) if ref.dim() == 3 else blended

    if verbose:
        ref, new = conds[keys[0]], out[keys[0]]
        delta = (new.float() - ref.float()).abs().max().item()
        print(f'[MCFM] mode={mode} variant={variant} window={offsets} '
              f'frames={len(keys)} shape={tuple(new.shape)}', flush=True)
        print(f'[MCFM] GATE-blend max|blended-vanilla| at frame {keys[0]} = '
              f'{delta:.5f}  (must be > 0, else the blend is a no-op)', flush=True)
        assert delta > 0, 'GATE-blend FAILED: blended tokens identical to vanilla'
    return out
