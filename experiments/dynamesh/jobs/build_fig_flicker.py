"""build_fig_flicker.py -- the duck figure in the FLICKER figure's format.

Replicates dynamesh_flicker.drawio / `colored_inset`: a left-aligned quoted-prompt title,
one full-bleed driving-video filmstrip with coloured ellipse markers, then per method a
row of full renders with circular zoom insets riding above them. Ring colours follow that
figure exactly: ours #66CC00, frozen #FF0000, and the strip markers match the ours ring.

WHAT THE INSET ZOOMS. The flicker figure zooms a fixed patch. Here the inset FOLLOWS THE
TRACKED DOT -- the same largest-blue-blob the frame choice was derived from -- so the
inset shows the same dot in all four panels and the drift reads at both scales. A fixed
patch would show a different dot each panel and prove nothing about motion.

CELLS ARE SQUARE, the flicker figure's were 385x559. That page framed a tall chair; a
wide duck in a 0.69 portrait cell is mostly white. Column count, gaps and the inset
offset are kept; only the aspect changes.
"""
import argparse, base64, io, math, xml.sax.saxutils as SU
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
LAB = 28

PAGE_W = 1830
TITLE = (10.0, 0.0, 1734.0, 112.0)
STRIP = (0.0, 130.0, 1820.0, 469.0)
FS_TITLE, FS_LABEL = 75, 61
MARGIN, GAP = 25.0, 40.0
CELL = (PAGE_W - 2 * MARGIN - 3 * GAP) / 4          # 415
XS = [MARGIN + i * (CELL + GAP) for i in range(4)]
INSET = 300.0
INSET_DX = -12.0
# The flicker figure's inset overlaps the top ~23% of its render. Reproduce the FRACTION,
# not its literal -162px: that offset was tuned for a 559px-tall cell and against a 415px
# square cell it pushed the insets up into the filmstrip's sprocket band.
INSET_OVERLAP_FRAC = 0.23
STRIP_GAP = 31.0                                    # clear air under the strip
RING = {'ours': '#66CC00', 'frozen': '#FF0000'}
ARM_LABEL = {'ours': 'DynaMesh (ours)', 'frozen': 'Frozen TRELLIS.2'}


def half(obj, view, idx, side):
    im = Image.open(E / f'out/view_{obj}_27m_{view}/frames/{idx:04d}.png').convert('RGB')
    w, h = im.size; s = h - LAB
    return im.crop((0, LAB, s, LAB + s) if side == 'frozen' else (s, LAB, 2 * s, LAB + s))


def track(im):
    """Centroid and radius of the largest blue blob, in pixels of `im`."""
    a = np.asarray(im).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > 90) & (b > r + 25) & (b > g + 15)
    lab, n = ndi.label(blue)
    if n == 0:
        w, h = im.size
        return (w / 2, h / 2), min(w, h) / 6
    ar = ndi.sum(blue, lab, range(1, n + 1))
    k = int(np.argmax(ar)) + 1
    cy, cx = ndi.center_of_mass(lab == k)
    return (float(cx), float(cy)), float(math.sqrt(ar.max() / math.pi))


def circle_inset(im, centre, rad, ring_hex, out_px=400, zoom=3.2, ring_px=22):
    """Circular crop centred on `centre`, ringed in `ring_hex`, transparent outside."""
    cx, cy = centre
    half_w = max(rad * zoom, out_px / 8)
    W, H = im.size
    half_w = min(half_w, W / 2, H / 2)
    cx = min(max(cx, half_w), W - half_w); cy = min(max(cy, half_w), H - half_w)
    crop = im.crop((int(cx - half_w), int(cy - half_w),
                    int(cx + half_w), int(cy + half_w))).resize((out_px, out_px), Image.LANCZOS)
    out = Image.new('RGBA', (out_px, out_px), (0, 0, 0, 0))
    mask = Image.new('L', (out_px, out_px), 0)
    ImageDraw.Draw(mask).ellipse([ring_px // 2, ring_px // 2,
                                  out_px - ring_px // 2, out_px - ring_px // 2], fill=255)
    out.paste(crop.convert('RGBA'), (0, 0), mask)
    d = ImageDraw.Draw(out)
    rgb = tuple(int(ring_hex[i:i + 2], 16) for i in (1, 3, 5))
    d.ellipse([ring_px // 2, ring_px // 2, out_px - ring_px // 2, out_px - ring_px // 2],
              outline=rgb + (255,), width=ring_px)
    return out


def build_strip(obj, idxs, ring_hex):
    """The full-bleed driving-video strip: border art + the four frames + dot markers."""
    border = Image.open(E / 'out/BOBDAILIES/filmstrip_border.jpeg').convert('RGB')
    SW, SH = 1820, 469
    strip = border.resize((SW, SH), Image.LANCZOS)
    # cell geometry, measured off the art and scaled to the new size
    cells = [(9, 313), (328, 632), (648, 951), (967, 1270)]
    cy0, cy1 = 52, 260
    sx, sy = SW / border.width, SH / border.height
    # CROP THE FRAMES TO THE SUBJECT before fitting them into the cells. Fitting the raw
    # 1440^2 frame leaves the duck occupying a third of the cell and the strip reads as
    # empty -- the same letterboxing that made the other figure look sparse. One shared
    # box across all four, so the subject never rescales between panels.
    gs = [Image.open(T2 / f'data/{obj}/frames_from_video/frame_{i:04d}.png').convert('RGB')
          for i in idxs]
    cw0 = int(cells[0][1] * sx) - int(cells[0][0] * sx)
    ch0 = int(cy1 * sy) - int(cy0 * sy)
    bb = box_union(gs, cw0 / ch0, pad=0.05)
    marks = []
    for k, (x0, x1) in enumerate(cells):
        cx0 = int(x0 * sx); ry0 = int(cy0 * sy)
        cw, ch = int(x1 * sx) - cx0, int(cy1 * sy) - ry0
        g = gs[k].crop(bb)
        c, rad = track(g)
        s = min(cw / g.width, ch / g.height)
        nw, nh = int(g.width * s), int(g.height * s)
        ox, oy = cx0 + (cw - nw) // 2, ry0 + (ch - nh) // 2
        strip.paste(g.resize((nw, nh), Image.LANCZOS), (ox, oy))
        px, py = ox + c[0] * s, oy + c[1] * s
        marks.append((STRIP[0] + px * STRIP[2] / SW, STRIP[1] + py * STRIP[3] / SH,
                      max(rad * s * 2.4, 46) * STRIP[2] / SW))
    b = io.BytesIO(); strip.save(b, 'JPEG', quality=92, optimize=True)
    return base64.b64encode(b.getvalue()).decode(), marks


def jpeg(im, q=92):
    b = io.BytesIO(); im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def png(im):
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def content_box(im, thresh=238):
    m = np.asarray(im.convert('L')) < thresh
    if not m.any(): return None
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1)
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_union(ims, aspect, pad=0.05):
    bs = [b for b in (content_box(i) for i in ims) if b]
    l = min(b[0] for b in bs); t = min(b[1] for b in bs)
    r = max(b[2] for b in bs); d = max(b[3] for b in bs)
    cx, cy = (l + r) / 2, (t + d) / 2
    hw = max((r - l) / 2, (d - t) / 2 * aspect) * (1 + pad); hh = hw / aspect
    W, H = ims[0].size
    hw = min(hw, W / 2); hh = min(hh, H / 2)
    cx = min(max(cx, hw), W - hw); cy = min(max(cy, hh), H - hh)
    box = (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))
    for im in ims:
        c = content_box(im)
        if c and (c[0] < box[0] or c[1] < box[1] or c[2] > box[2] or c[3] > box[3]):
            raise SystemExit(f'FAILED: strip crop {box} clips content {c}')
    return box


def square_union(ims, pad=0.02):
    bs = [b for b in (content_box(i) for i in ims) if b]
    l = min(b[0] for b in bs); t = min(b[1] for b in bs)
    r = max(b[2] for b in bs); d = max(b[3] for b in bs)
    cx, cy = (l + r) / 2, (t + d) / 2
    hh = max(r - l, d - t) * (1 + pad) / 2
    W, H = ims[0].size
    hh = min(hh, W / 2, H / 2)
    cx = min(max(cx, hh), W - hh); cy = min(max(cy, hh), H - hh)
    box = (int(cx - hh), int(cy - hh), int(cx + hh), int(cy + hh))
    for im in ims:
        c = content_box(im)
        if c and (c[0] < box[0] or c[1] < box[1] or c[2] > box[2] or c[3] > box[3]):
            raise SystemExit(f'FAILED: crop {box} clips content {c}')
    return box


IMGSTY = ('shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;'
          'verticalAlign=top;imageAspect=0;image=data:image/{t},{b};')
INSETSTY = IMGSTY + 'imageBorder=none;strokeWidth=1;imageBackground=none;'
TXTSTY = ('text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={a};'
          'verticalAlign=middle;rounded=0;fontSize={fs};fontFamily={f};fontColor=#000000;')
ELLSTY = ('ellipse;whiteSpace=wrap;html=1;shapeInside=1;fillColor=none;strokeWidth=8;'
          'strokeColor=light-dark({c},{c});')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--view', required=True)
    ap.add_argument('--view2', default=None,
                    help='second camera; given, the page becomes two plain render rows '
                         '(the earlier layout) at THIS figure\'s scale, with no insets')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--arm', default='ours', choices=['ours', 'frozen'])
    ap.add_argument('--name', required=True)
    A = ap.parse_args()
    idxs = [int(x) for x in A.frames.split(',')]
    ring = RING[A.arm]

    strip_b64, marks = build_strip(A.obj, idxs, ring)

    def row_cells(view):
        ims = [half(A.obj, view, i, A.arm) for i in idxs]
        bb = square_union(ims)
        return [x.crop(bb) for x in ims]

    cells = row_cells(A.view)
    cells2 = row_cells(A.view2) if A.view2 else None
    insets = []
    if not A.view2:
        for cell in cells:
            c, rad = track(cell)
            insets.append(circle_inset(cell, c, rad, ring))

    if A.view2:
        y_render = STRIP[1] + STRIP[3] + GAP
        y_render2 = y_render + CELL + GAP
        page_h = int(y_render2 + CELL + GAP)
        y_inset = None
    else:
        y_inset = STRIP[1] + STRIP[3] + STRIP_GAP
        y_render = y_inset + INSET - INSET_OVERLAP_FRAC * CELL
        page_h = int(y_render + CELL + 50)
        assert y_inset >= STRIP[1] + STRIP[3], 'inset would overlap the filmstrip'

    out, n = [], [0]
    def cell_xml(x, y, w, h, style, value=''):
        n[0] += 1
        v = SU.escape(value, {'"': '&quot;', "'": '&apos;'})
        out.append(f'<mxCell id="k{n[0]}" value="{v}" style="{style}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    cell_xml(*TITLE, TXTSTY.format(a='left', fs=FS_TITLE, f='Comic Sans MS'), A.title)
    cell_xml(*STRIP, IMGSTY.format(t='jpeg', b=strip_b64))
    if not A.view2:
        for mx, my, mr in marks:
            cell_xml(mx - mr, my - mr, mr * 2, mr * 2, ELLSTY.format(c=ring))
    for i, x in enumerate(XS):
        cell_xml(x, y_render, CELL, CELL, IMGSTY.format(t='jpeg', b=jpeg(cells[i])))
    if cells2:
        for i, x in enumerate(XS):
            cell_xml(x, y_render2, CELL, CELL, IMGSTY.format(t='jpeg', b=jpeg(cells2[i])))
    else:
        for i, x in enumerate(XS):
            cell_xml(x + INSET_DX, y_inset, INSET, INSET,
                     INSETSTY.format(t='png', b=png(insets[i])))

    xml = (f'<mxfile host="app.diagrams.net">\n<diagram name="{A.name}">\n'
           f'<mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" page="1" '
           f'pageWidth="{PAGE_W}" pageHeight="{page_h}" math="0" shadow="0">\n<root>\n'
           f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(out) +
           '\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n')
    o = E / 'out/BOBDAILIES' / f'{A.name}.drawio.xml'
    o.write_text(xml)
    print(f'page {PAGE_W}x{page_h}  cells {CELL:.0f}px  inset {INSET:.0f}px  ring {ring}')
    print(f'  {o}  {o.stat().st_size/1048576:.2f} MB')


if __name__ == '__main__':
    main()
