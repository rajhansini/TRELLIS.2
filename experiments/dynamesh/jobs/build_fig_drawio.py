"""build_fig_drawio.py -- the two-camera comparison figure, as BOTH a reviewable HTML
artifact and a draw.io page, from ONE set of layout constants.

Structure follows dynamesh_failure.drawio / better_second_row_yaw: a quoted-prompt title,
a filmstrip of driving-video frames, then the result at TWO cameras -- an upright row and
a row tilted 6 degrees.

--arms ours|frozen|both
  A single-arm page carries NO method headline: with one block on the page the label is
  redundant and costs a 110px band that is better spent on the renders. 'both' restores
  the headline because then it is load-bearing.

SIZING. Cards are computed to FILL the page width rather than copied from the source
page, whose 205x314 slots were sized for a tall vase and left a wide subject swimming in
white. Four cards + three gaps + two margins = page width, exactly.

A ROTATED CARD IS TALLER THAN ITS BOX: draw.io rotates about the geometry centre, so a
square of side w at t degrees occupies w*(cos t + sin t) vertically. Laying the stack out
with the unrotated height is what once put the tilted row through the label below it.
Every y is DERIVED, so changing card size or tilt cannot reintroduce that overlap.
"""
import argparse, base64, io, math, xml.sax.saxutils as SU
from pathlib import Path
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
LAB = 28                                   # label bar height on every render frame

PAGE_W = 1340
MARGIN, GAP, MARGIN_TOP = 18.0, 10.0, 40.0
CARD = (PAGE_W - 2 * MARGIN - 3 * GAP) / 4
XS = [MARGIN + i * (CARD + GAP) for i in range(4)]
# MEASURED off filmstrip_border.jpeg, not inherited: its cells are 309.1 x 211.9 page
# units, centred at these x. The source page dropped 200px SQUARE frames into those
# landscape cells, which letterboxed a wide subject and left the strip looking empty.
# Filling the cell is the single biggest whitespace win on the page.
GT_W, GT_H = 300.0, 206.0
XS_GT = [c - GT_W / 2 for c in (194.7, 519.6, 845.1, 1170.0)]
GT_DY = 158.9 - GT_H / 2                            # cell centre, relative to strip top
H_STRIP, H_TITLE, H_LABEL = 314.75, 102.0, 94.25
# TILT defaults to 0: of the eight paper figures, only dynamesh_failure's final tab
# rotates its second row. The other seven are axis-aligned, and the rotation bounding
# box also costs ~30px of vertical air per row -- the whitespace we are trying to spend
# on bigger renders. --tilt 6 restores the failure figure's look.
TILT, OVERLAP = 0, 9.0
GAP_BLOCK, GAP_LABEL = 24.0, 14.0
FONT_TITLE, FONT_HEAD, FS = 'Comic Sans MS', 'Helvetica', 65
ARM_LABEL = {'ours': 'DynaMesh (ours)', 'frozen': 'Frozen TRELLIS.2'}


def rot_h(w, deg):
    r = math.radians(deg)
    return w * (abs(math.cos(r)) + abs(math.sin(r)))


def layout(arms):
    y = MARGIN_TOP
    L = {'title': (MARGIN, y, PAGE_W - 2 * MARGIN, H_TITLE)}
    y += H_TITLE + GAP_BLOCK
    L['strip'] = (MARGIN, y, 1303.81, H_STRIP)
    L['gt_y'] = y + GT_DY
    y += H_STRIP + GAP_BLOCK
    for arm in arms:
        if len(arms) > 1:                          # headline only when it disambiguates
            lw = 726.23
            L[f'{arm}_label'] = ((PAGE_W - lw) / 2, y, lw, H_LABEL)
            y += H_LABEL + GAP_LABEL
        L[f'{arm}_up_y'] = y
        y += CARD + (GAP if not TILT else -OVERLAP)
        L[f'{arm}_tilt_y'] = y
        y = y + CARD / 2 + rot_h(CARD, TILT) / 2 + GAP_BLOCK
    return L, int(round(y - GAP_BLOCK + MARGIN_TOP))


def jpeg(im, q=90):
    b = io.BytesIO(); im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def half(obj, view, idx, side):
    p = E / f'out/view_{obj}_27m_{view}/frames/{idx:04d}.png'
    im = Image.open(p).convert('RGB'); w, h = im.size
    s = h - LAB
    return im.crop((0, LAB, s, LAB + s) if side == 'frozen' else (s, LAB, 2 * s, LAB + s))


def content_box(im, thresh=238):
    """Bounding box of the LARGEST connected blob, not of every non-white pixel.

    A plain getbbox() over "anything darker than white" is defeated by a few stray
    near-white speckles at the frame edge: one such frame reported a full-width box,
    which poisoned the shared crop and sliced ~96px off the subject in every frame of
    the row. Taking the biggest component ignores speckle by construction.
    """
    import numpy as np
    from scipy import ndimage as ndi
    m = np.asarray(im.convert('L')) < thresh
    if not m.any():
        return None
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_union(ims, aspect, pad):
    """Shared crop of a given w:h aspect containing every frame's content.

    The square version letterboxes a wide subject; the film cells are 1.46:1, so the
    reference frames get a matching box and fill them.
    """
    bs = [b for b in (content_box(i) for i in ims) if b]
    l = min(b[0] for b in bs); t = min(b[1] for b in bs)
    r = max(b[2] for b in bs); d = max(b[3] for b in bs)
    cx, cy = (l + r) / 2, (t + d) / 2
    hw = max((r - l) / 2, (d - t) / 2 * aspect) * (1 + pad)
    hh = hw / aspect
    W, H = ims[0].size
    hw = min(hw, W / 2); hh = min(hh, H / 2)
    cx = min(max(cx, hw), W - hw); cy = min(max(cy, hh), H - hh)
    box = (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))
    for im in ims:
        c = content_box(im)
        if c and (c[0] < box[0] or c[1] < box[1] or c[2] > box[2] or c[3] > box[3]):
            raise SystemExit(f'FAILED: crop {box} clips content {c}')
    return box


def square_union(ims, pad):
    """ONE square crop shared by a whole row, so the subject never rescales between
    timesteps -- that would read as motion the method is not producing."""
    bs = [b for b in (content_box(i) for i in ims) if b]
    l = min(b[0] for b in bs); t = min(b[1] for b in bs)
    r = max(b[2] for b in bs); d = max(b[3] for b in bs)
    cx, cy = (l + r) / 2, (t + d) / 2
    hh = max(r - l, d - t) * (1 + pad) / 2
    W, H = ims[0].size
    hh = min(hh, W / 2, H / 2)                    # can't be wider than the frame
    # SHIFT the square inside the frame; do NOT shrink it. Shrinking is what cut the
    # subject: the box stayed centred on the content and simply lost its edges.
    cx = min(max(cx, hh), W - hh)
    cy = min(max(cy, hh), H - hh)
    box = (int(cx - hh), int(cy - hh), int(cx + hh), int(cy + hh))
    # GATE: the crop must contain every frame's content. Silent clipping is exactly the
    # failure this function exists to prevent.
    for im in ims:
        c = content_box(im)
        if c and (c[0] < box[0] or c[1] < box[1] or c[2] > box[2] or c[3] > box[3]):
            raise SystemExit(f'FAILED: crop {box} clips content {c} -- would cut the subject')
    return box


def collect(obj, v_up, v_tilt, idxs, arms, px=620):
    r = {}
    g = [Image.open(T2 / f'data/{obj}/frames_from_video/frame_{i:04d}.png').convert('RGB')
         for i in idxs]
    bb = box_union(g, GT_W / GT_H, pad=0.04)
    r['gt'] = [jpeg(x.crop(bb).resize((int(GT_W * 2), int(GT_H * 2)), Image.LANCZOS)) for x in g]
    for tag, v in (('up', v_up), ('tilt', v_tilt)):
        # both arms share ONE box per camera: different crops would draw the two methods
        # at different sizes and the comparison would be unreadable
        pair = {a: [half(obj, v, i, a) for i in idxs] for a in ('ours', 'frozen')}
        bb = square_union(pair['ours'] + pair['frozen'], pad=0.02)
        for a in arms:
            r[f'{a}_{tag}'] = [jpeg(x.crop(bb).resize((px, px), Image.LANCZOS))
                               for x in pair[a]]
    return r


IMGSTY = ('shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;'
          'verticalAlign=top;imageAspect=0;image=data:image/jpeg,{b}{rot}')
TXTSTY = ('text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={a};'
          'verticalAlign=top;rounded=0;fontSize=%d;fontFamily={f};fontColor=#000000;' % FS)


def drawio(img, title, strip, arms, L, page_h, page='fig'):
    out, n = [], [0]

    def cell(x, y, w, h, style, value=''):
        n[0] += 1
        v = SU.escape(value, {'"': '&quot;', "'": '&apos;'})
        out.append(f'<mxCell id="f{n[0]}" value="{v}" style="{style}" vertex="1" '
                   f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
                   f'as="geometry"/></mxCell>')
    cell(*L['title'], TXTSTY.format(a='left', f=FONT_TITLE), title)
    cell(*L['strip'], IMGSTY.format(b=strip, rot=''))
    for i, x in enumerate(XS_GT):
        cell(x, L['gt_y'], GT_W, GT_H, IMGSTY.format(b=img['gt'][i], rot=''))
    for arm in arms:
        if f'{arm}_label' in L:
            cell(*L[f'{arm}_label'], TXTSTY.format(a='center', f=FONT_HEAD), ARM_LABEL[arm])
        for i, x in enumerate(XS):
            cell(x, L[f'{arm}_up_y'], CARD, CARD, IMGSTY.format(b=img[f'{arm}_up'][i], rot=''))
        for i, x in enumerate(XS):
            cell(x, L[f'{arm}_tilt_y'], CARD, CARD,
                 IMGSTY.format(b=img[f'{arm}_tilt'][i], rot=f';rotation={TILT};'))
    return (f'<mxfile host="app.diagrams.net">\n<diagram name="{page}">\n'
            f'<mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" page="1" '
            f'pageWidth="{PAGE_W}" pageHeight="{page_h}" math="0" shadow="0">\n<root>\n'
            f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(out) +
            '\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n')


def html(img, title, strip, arms, L, page_h, cap_up, cap_tilt, subtitle):
    def im(x, y, w, h, b, rot=0):
        t = f' transform:rotate({rot}deg);' if rot else ''
        return (f'<img src="data:image/jpeg;base64,{b}" style="left:{x}px;top:{y}px;'
                f'width:{w}px;height:{h}px;{t}">')
    def tx(box, txt, fam, align):
        x, y, w, h = box
        return (f'<div class="t" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
                f'font-family:\'{fam}\',cursive,sans-serif;text-align:{align}">{txt}</div>')
    P = [tx(L['title'], SU.escape(title), FONT_TITLE, 'left'), im(*L['strip'], strip)]
    P += [im(x, L['gt_y'], GT_W, GT_H, img['gt'][i]) for i, x in enumerate(XS_GT)]
    for arm in arms:
        if f'{arm}_label' in L:
            P.append(tx(L[f'{arm}_label'], ARM_LABEL[arm], FONT_HEAD, 'center'))
        P += [im(x, L[f'{arm}_up_y'], CARD, CARD, img[f'{arm}_up'][i]) for i, x in enumerate(XS)]
        P += [im(x, L[f'{arm}_tilt_y'], CARD, CARD, img[f'{arm}_tilt'][i], TILT)
              for i, x in enumerate(XS)]
    stage = '\n'.join(P)
    who = ' + '.join(ARM_LABEL[a] for a in arms)
    return f"""<title>Duck Figure Draft</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{{--bg:#f7f5f1;--surface:#fff;--border:#ddd9d0;--text:#1c1d22;--text-dim:#5c5f6b;--accent:#b9762c;--mono:'IBM Plex Mono',ui-monospace,monospace;--sans:'IBM Plex Sans',-apple-system,sans-serif;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#15161b;--surface:#1c1e25;--border:#34384a;--text:#eceef2;--text-dim:#9195a3;--accent:#d9974f;}}}}
:root[data-theme="dark"]{{--bg:#15161b;--surface:#1c1e25;--border:#34384a;--text:#eceef2;--text-dim:#9195a3;--accent:#d9974f;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.5}}
header{{padding:32px 28px 20px;border-bottom:1px solid var(--border);background:var(--surface)}}
header .in{{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:6px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:600}}
h1{{font-size:26px;margin:0;letter-spacing:-.01em}}
.sub{{color:var(--text-dim);font-size:14.5px;margin:0;max-width:74ch}}
main{{max-width:1180px;margin:0 auto;padding:24px 28px 72px}}
.scroll{{overflow-x:auto;border:1px solid var(--border);border-radius:10px;background:#fff}}
.page{{position:relative;width:{PAGE_W}px;height:{page_h}px;background:#fff;transform-origin:top left}}
.page img{{position:absolute;object-fit:contain}}
.page .t{{position:absolute;font-size:{FS}px;color:#000;line-height:1.08;white-space:nowrap}}
.legend{{margin-top:20px;display:flex;flex-direction:column;gap:8px;font-size:14px;color:var(--text-dim)}}
.legend b{{color:var(--text)}} code{{font-family:var(--mono);font-size:12.5px}}
</style>
<header><div class="in">
  <span class="eyebrow">3DV &apos;27 &middot; DynaMesh &middot; figure draft</span>
  <h1>{who} &mdash; two cameras</h1>
  <p class="sub">{SU.escape(subtitle)}</p>
</div></header>
<main>
  <div class="scroll"><div class="page" id="pg">{stage}</div></div>
  <div class="legend">
    <div><b>Row 1</b> &mdash; {SU.escape(cap_up)}</div>
    <div><b>Row 2</b> &mdash; {SU.escape(cap_tilt)}, tilted {TILT}&deg;</div>
    <div><b>Filmstrip</b> &mdash; the driving clip, the only camera the loss ever saw.</div>
    <div>Page <code>{PAGE_W}&times;{page_h}</code>, cards <code>{CARD:.0f}px</code> computed to fill the width.</div>
  </div>
</main>
<script>
function fit(){{const p=document.getElementById('pg'),w=p.parentElement.clientWidth;
  const s=Math.min(1,w/{PAGE_W});p.style.transform='scale('+s+')';
  p.parentElement.style.height=({page_h}*s)+'px';}}
fit();addEventListener('resize',fit);
</script>"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--view-up', required=True)
    ap.add_argument('--view-tilt', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--arms', default='ours', choices=['ours', 'frozen', 'both'])
    ap.add_argument('--cap-up', default=''); ap.add_argument('--cap-tilt', default='')
    ap.add_argument('--sub', default=''); ap.add_argument('--name', required=True)
    ap.add_argument('--tilt', type=float, default=None,
                    help='second-row rotation in degrees; default 0 (paper convention)')
    A = ap.parse_args()
    if A.tilt is not None:
        globals()['TILT'] = A.tilt
    arms = ['ours', 'frozen'] if A.arms == 'both' else [A.arms]
    idxs = [int(x) for x in A.frames.split(',')]
    L, PH = layout(arms)
    strip = base64.b64encode((E / 'out/BOBDAILIES/filmstrip_border.jpeg').read_bytes()).decode()
    img = collect(A.obj, A.view_up, A.view_tilt, idxs, arms)
    o = E / 'out/BOBDAILIES'
    (o / f'{A.name}.drawio.xml').write_text(drawio(img, A.title, strip, arms, L, PH, A.name))
    (o / f'{A.name}.html').write_text(
        html(img, A.title, strip, arms, L, PH, A.cap_up, A.cap_tilt, A.sub))
    print(f'page {PAGE_W}x{PH}  cards {CARD:.1f}px  arms={arms}')
    for f in (f'{A.name}.drawio.xml', f'{A.name}.html'):
        print(f'  {o/f}  {(o/f).stat().st_size/1048576:.2f} MB')
