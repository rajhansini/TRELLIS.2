# ═════════════════════════ schematic chain, 8 elements ════════════════════════
W, H = 1460, 546
S = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
     f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">', DEFS,
     f'<rect width="{W}" height="{H}" fill="{PAPER}"/>']

CY   = 248      # main chain
YTOP = 118      # token lane
YBOT = 352      # voxel-token lane
VW   = 212      # every voxel structure is drawn at the same size, same view

def block(x, y, w, h, t1, t2, subs, ptxt, pf, pe, pt, fill, edge, tcol):
    o = [rbox(x, y, w, h, fill, edge, rx=9, sw=1.8)]
    o.append(txt(x + w / 2, y + 26, t1, size=16, weight='700', fill=tcol))
    o.append(txt(x + w / 2, y + 43, t2, size=16, weight='700', fill=tcol))
    for n, t in enumerate(subs):
        o.append(txt(x + w / 2, y + 65 + n * 14, t, size=12.5, fill=tcol, op=0.80))
    o.append(pill(x + 8, y + h - 30, w - 16, 19, ptxt, pf, pe, pt, size=12))
    return '\n'.join(o)

def caption(cx, y, t1, t2, t3=None):
    o = (txt(cx, y, t1, size=20, weight='700', fill=INK) + '\n'
         + txt(cx, y + 19, t2, size=15, fill=INK_SOFT))
    if t3:
        o += '\n' + txt(cx, y + 36, t3, size=15, fill=INK_SOFT)
    return o

# ── 1  three token maps ───────────────────────────────────────────────────────
FX = [48, 158, 268]; FS = 100
for n, l in enumerate(['f − 1', 'f', 'f + 1']):
    S.append(grid_svg(FX[n], YTOP - FS / 2, FS, frame_grid(n)))
    S.append(txt(FX[n] + FS / 2, YTOP - FS / 2 - 10, l, size=16, weight='700', fill=hexc(TINTS[n])))
S.append(caption((FX[0] + FX[2] + FS) / 2, YTOP + FS / 2 + 27, 'token maps',
                 'one per frame of the window'))

# ── 2  temporal attention ─────────────────────────────────────────────────────
TX, TW, TH = 400, 108, 122
S.append(arrow(FX[2] + FS + 4, YTOP, TX - 6, YTOP, sw=2.2))
S.append(block(TX, YTOP - TH / 2, TW, TH, 'TEMPORAL', 'ATTENTION', ['over the window'],
               'no parameters', '#FFFFFF', FREE_EDGE, '#B4650B', FREE_FILL, FREE_EDGE, '#B4650B'))

# ── 3  blended tokens ─────────────────────────────────────────────────────────
GX, GS = 540, 120
S.append(arrow(TX + TW + 4, YTOP, GX - 6, YTOP, sw=2.2))
S.append(grid_svg(GX, YTOP - GS / 2, GS, blend_grid(), lw=0.5))
S.append(caption(GX + GS / 2, YTOP + GS / 2 + 27, 'blended tokens', 'one map, mixed over the window'))

# ── 5  cross-attention (two inputs converge here) ─────────────────────────────
XA, BW, BH = 722, 98, 152
S.append(block(XA, CY - BH / 2, BW, BH, 'CROSS-', 'ATTENTION',
               ['Q ← voxels', 'K, V ← tokens'], 'LoRA  r = 4',
               LORA_FILL, LORA_EDGE, '#1E7A44', BOX_FILL, BOX_EDGE, '#23417E'))

# ── 4  voxel tokens: transparent voxel structure, second input ────────────────
VTX = 108
gT, hT, wT = voxel_svg(VTX, YBOT - 105, VW, 'glass')
S.append(gT)
S.append(caption(VTX + VW / 2, YBOT + 128, 'voxel tokens', 'shape latent + noise',
                 'a sparse voxel structure, interiors transparent'))

S.append(curve(GX + GS + 5, YTOP, GX + GS + 56, YTOP, XA - 58, CY - 42,
               XA - 7, CY - 42, sw=2.2))
S.append(curve(VTX + VW + 5, YBOT - 14, VTX + VW + 150, YBOT - 14, XA - 120, CY + 42,
               XA - 7, CY + 42, sw=2.2))

# ── 6  voxels after cross-attention ───────────────────────────────────────────
VA_X = 846
gA, hA, wA = voxel_svg(VA_X, CY - 105, VW, 'ca')
S.append(arrow(XA + BW + 4, CY, VA_X - 8, CY, sw=2.2))
S.append(gA)

# ── 7  self-attention ─────────────────────────────────────────────────────────
XS = 1082
S.append(arrow(VA_X + VW + 8, CY, XS - 6, CY, sw=2.2))
S.append(block(XS, CY - BH / 2, BW, BH, 'SELF-', 'ATTENTION', ['Q, K, V ← voxels'],
               'LoRA  r = 4', LORA_FILL, LORA_EDGE, '#1E7A44', BOX_FILL, BOX_EDGE, '#23417E'))

# ── 8  voxels after self-attention ────────────────────────────────────────────
VB_X = 1204
gB, hB, wB = voxel_svg(VB_X, CY - 105, VW, 'sa')
S.append(arrow(XS + BW + 4, CY, VB_X - 8, CY, sw=2.2))
S.append(gB)

VCAP = CY - 105 + hA + 30
S.append(caption(VA_X + VW / 2, VCAP, 'after cross-attention',
                 'response on the front — head and forelegs'))
S.append(caption(VB_X + VW / 2, VCAP, 'after self-attention',
                 'carried through the whole structure'))

S.append('</svg>')
OUT_SVG = pathlib.Path(__file__).resolve().parent / 'fig_mcfm_rung27.svg'
OUT_SVG.write_text('\n'.join(S))
print('wrote', OUT_SVG, W, H, 'voxel h=%.0f' % hA)
