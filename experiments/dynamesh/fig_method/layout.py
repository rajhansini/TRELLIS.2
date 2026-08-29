# ═══════════════ two-row serpentine, matching the system figure's reading order ═
#   row 1   blended tokens  <--  TEMPORAL ATTENTION  <--  f-1  f  f+1
#                  |
#                  v   (blended tokens sits directly above cross-attention)
#   row 2   voxel tokens --> CROSS-ATTN --> after CA --> SELF-ATTN --> after SA
#
# Row 1 runs right-to-left and row 2 left-to-right, so the eye turns once and
# lands on the block the blended tokens actually feed. The old single-row version
# needed two long curves to reach cross-attention from opposite sides.
W, H = 1168, 612
S = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
     f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS,
     f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

CY1 = 132       # token lane
CY2 = 400       # voxel lane
VW  = 212       # every voxel structure: same size, same view, same cells

def block(x, y, w, h, t1, t2, subs, ptxt, pf, pe, pt, fill, edge, tcol):
    o = [rbox(x, y, w, h, fill, edge, rx=9, sw=1.8)]
    o.append(txt(x + w / 2, y + 26, t1, size=15.5, weight='700', fill=tcol))
    o.append(txt(x + w / 2, y + 43, t2, size=15.5, weight='700', fill=tcol))
    for n, t in enumerate(subs):
        o.append(txt(x + w / 2, y + 66 + n * 14.5, t, size=11.5, fill=tcol, op=0.82))
    o.append(pill(x + 8, y + h - 29, w - 16, 19, ptxt, pf, pe, pt, size=11.5))
    return '\n'.join(o)

def caption(cx, y, *lines, anchor='middle'):
    o = [txt(cx, y, lines[0], size=19, weight='700', fill=INK, anchor=anchor)]
    for n, t in enumerate(lines[1:]):
        o.append(txt(cx, y + 19 + n * 17, t, size=14, fill=INK_SOFT, anchor=anchor))
    return '\n'.join(o)

# ══ ROW 2 first: it sets the width, and row 1 hangs off cross-attention ════════
VT_X, XA, VA_X, XS, VB_X = 44, 300, 478, 734, 912
BW, BH = 140, 150

# ── voxel tokens ──────────────────────────────────────────────────────────────
gT, hT, wT = voxel_svg(VT_X, CY2 - 108, VW, 'glass')
S.append(gT)

# ── cross-attention ───────────────────────────────────────────────────────────
S.append(block(XA, CY2 - BH / 2, BW, BH, 'CROSS-', 'ATTENTION',
               ['Q ← voxels', 'K, V ← 1,029 image tokens', 'to_q,  to_kv,  to_out'],
               'LoRA  r = 4, α = 4',
               LORA_FILL, LORA_EDGE, '#1E7A44', BOX_FILL, BOX_EDGE, '#23417E'))
S.append(arrow(VT_X + VW + 6, CY2, XA - 7, CY2, sw=2.2))

# ── voxels after cross-attention ──────────────────────────────────────────────
gA, hA, wA = voxel_svg(VA_X, CY2 - 108, VW, 'ca')
S.append(arrow(XA + BW + 5, CY2, VA_X - 8, CY2, sw=2.2))
S.append(gA)

# ── self-attention ────────────────────────────────────────────────────────────
S.append(arrow(VA_X + VW + 8, CY2, XS - 7, CY2, sw=2.2))
S.append(block(XS, CY2 - BH / 2, BW, BH, 'SELF-', 'ATTENTION',
               ['Q, K, V ← voxels', 'sa_qkv  (fused Q, K, V)', 'sa_out'],
               'LoRA  r = 4, α = 4',
               LORA_FILL, LORA_EDGE, '#1E7A44', BOX_FILL, BOX_EDGE, '#23417E'))

# ── voxels after self-attention ───────────────────────────────────────────────
gB, hB, wB = voxel_svg(VB_X, CY2 - 108, VW, 'sa')
S.append(arrow(XS + BW + 5, CY2, VB_X - 8, CY2, sw=2.2))
S.append(gB)

VCAP = CY2 - 108 + hA + 28
S.append(caption(VT_X + VW / 2, VCAP, 'voxel tokens',
                 'shape latent + noise', 'the same 484 cells in every panel'))
S.append(caption(VA_X + VW / 2, VCAP, 'after cross-attention',
                 'response on the front — head and forelegs'))
S.append(caption(VB_X + VW / 2, VCAP, 'after self-attention',
                 'carried through the whole structure'))

# ══ ROW 1: right-to-left, ending directly above cross-attention ═══════════════
GS = 124
GX = XA + BW / 2 - GS / 2
TW, TH = 124, 132
TX = GX + GS + 16
FS, FGAP = 124, 14
FX = [TX + TW + 16 + n * (FS + FGAP) for n in range(3)]

for n, l in enumerate(['f − 1', 'f', 'f + 1']):
    S.append(grid_svg(FX[n], CY1 - FS / 2, FS, frame_grid(n)))
    S.append(txt(FX[n] + FS / 2, CY1 - FS / 2 - 11, l, size=16, weight='700',
                 fill=hexc(TINTS[n])))
S.append(caption((FX[0] + FX[2] + FS) / 2, CY1 + FS / 2 + 26,
                 'token maps', 'one per frame of the window'))

S.append(arrow(FX[0] - 6, CY1, TX + TW + 6, CY1, sw=2.2))
S.append(block(TX, CY1 - TH / 2, TW, TH, 'TEMPORAL', 'ATTENTION',
               ['per token position', 'softmax over W = 3', 'no projections'],
               'no parameters',
               '#FFFFFF', FREE_EDGE, '#B4650B', FREE_FILL, FREE_EDGE, '#B4650B'))

S.append(arrow(TX - 6, CY1, GX + GS + 6, CY1, sw=2.2))
S.append(grid_svg(GX, CY1 - GS / 2, GS, blend_grid(), lw=0.5))
S.append(caption(GX - 18, CY1 - 6, 'blended tokens',
                 'one map, mixed', 'over the window', anchor='end'))

# ── the turn: straight down into cross-attention ──────────────────────────────
S.append(arrow(GX + GS / 2, CY1 + GS / 2 + 6, GX + GS / 2, CY2 - BH / 2 - 8, sw=2.2))

# ── footnote: what "LoRA r = 4" and "no parameters" actually cost ─────────────
S.append(txt(W / 2, H - 20,
             'LoRA: 5 adapters \u00d7 30 blocks = 2,334,720 trainable parameters, '
             '0.18% of the 1.3B texture flow.   Temporal attention adds none.',
             size=13, fill=INK_SOFT, op=0.95))

S.append('</svg>')
OUT_SVG = pathlib.Path(__file__).resolve().parent / 'fig_mcfm_rung27.svg'
OUT_SVG.write_text('\n'.join(S))
print('wrote', OUT_SVG, W, H, 'voxel h=%.0f' % hA)
