"""build_fig_trimmed.py -- the duck figure at dynamesh_flicker_trimmed's exact size.

Page 1243x546, side by side: DynaMesh (ours) left, Frozen TRELLIS.2 right, a centred
driving-video strip above both with ring markers, a dashed divider between the blocks.
Every coordinate below is read off `colored_inset` in that file, not re-derived.

CELLS ARE SQUARE. The source cells are 124.5x180.8 -- portrait, sized for a chair. A wide
duck letterboxes in that, so height follows width here. Page size, column x positions,
inset size, label and title typography, strip box and divider are all untouched.

The insets follow the tracked dot, so the same blue dot appears in all four and its drift
reads at both scales; the strip markers point at that dot in the driving video.
"""
import argparse, base64, io, math, xml.sax.saxutils as SU
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
LAB = 28

PAGE_W, PAGE_H = 1243, 546
TITLE = (327.2, 0.0, 588.6, 58.0)
STRIP = (327.2, 62.1, 588.6, 151.7)
FS = 44
LABEL_H = 58.0
LABEL_Y = 227.8
BLOCK = {                       # per method: label box, inset xs, render xs
    'ours':   dict(label=(20.0, 570.5), insets=[20.0, 167.5, 311.7, 460.5],
                   renders=[27.8, 172.0, 316.6, 464.4], ring='#66CC00'),
    'frozen': dict(label=(652.5, 570.5), insets=[652.5, 800.0, 944.2, 1093.0],
                   renders=[656.4, 803.9, 948.4, 1096.9], ring='#FF0000'),
}
INSET, INSET_Y = 97.0, 292.8
CELL_W, RENDER_Y = 124.5, 346.8
DIVIDER = (621.5, 223.793, 621.5, 533.208)
ARM_LABEL = {'ours': 'DynaMesh (ours)', 'frozen': 'Frozen&nbsp;TRELLIS.2'}


def half(obj, view, idx, side):
    im = Image.open(E / f'out/view_{obj}_27m_{view}/frames/{idx:04d}.png').convert('RGB')
    w, h = im.size; s = h - LAB
    return im.crop((0, LAB, s, LAB + s) if side == 'frozen' else (s, LAB, 2 * s, LAB + s))


def content_box(im, thresh=238):
    m = np.asarray(im.convert('L')) < thresh
    if not m.any(): return None
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1)
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_union(ims, aspect, pad):
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
            raise SystemExit(f'FAILED: crop {box} clips content {c}')
    return box


def track(im):
    a = np.asarray(im).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blue = (b > 90) & (b > r + 25) & (b > g + 15)
    lab, n = ndi.label(blue)
    if n == 0:
        w, h = im.size; return (w / 2, h / 2), min(w, h) / 6
    ar = ndi.sum(blue, lab, range(1, n + 1)); k = int(np.argmax(ar)) + 1
    cy, cx = ndi.center_of_mass(lab == k)
    return (float(cx), float(cy)), float(math.sqrt(ar.max() / math.pi))


def circle_inset(im, centre, rad, ring_hex, out_px=320, zoom=3.2, ring_px=18):
    cx, cy = centre
    hw = max(rad * zoom, out_px / 10); W, H = im.size
    hw = min(hw, W / 2, H / 2)
    cx = min(max(cx, hw), W - hw); cy = min(max(cy, hw), H - hw)
    crop = im.crop((int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw))) \
             .resize((out_px, out_px), Image.LANCZOS)
    out = Image.new('RGBA', (out_px, out_px), (0, 0, 0, 0))
    mask = Image.new('L', (out_px, out_px), 0)
    ImageDraw.Draw(mask).ellipse([ring_px // 2, ring_px // 2,
                                  out_px - ring_px // 2, out_px - ring_px // 2], fill=255)
    out.paste(crop.convert('RGBA'), (0, 0), mask)
    rgb = tuple(int(ring_hex[i:i + 2], 16) for i in (1, 3, 5))
    ImageDraw.Draw(out).ellipse([ring_px // 2, ring_px // 2,
                                 out_px - ring_px // 2, out_px - ring_px // 2],
                                outline=rgb + (255,), width=ring_px)
    return out


def build_strip(obj, idxs):
    border = Image.open(E / 'out/BOBDAILIES/filmstrip_border.jpeg').convert('RGB')
    SW, SH = 1180, 304                                   # ~ STRIP box at 2x
    strip = border.resize((SW, SH), Image.LANCZOS)
    cells = [(9, 313), (328, 632), (648, 951), (967, 1270)]
    cy0, cy1 = 52, 260
    sx, sy = SW / border.width, SH / border.height
    gs = [Image.open(T2 / f'data/{obj}/frames_from_video/frame_{i:04d}.png').convert('RGB')
          for i in idxs]
    cw0 = int(cells[0][1] * sx) - int(cells[0][0] * sx)
    ch0 = int(cy1 * sy) - int(cy0 * sy)
    bb = box_union(gs, cw0 / ch0, pad=0.05)
    marks = []
    for k, (x0, x1) in enumerate(cells):
        cx0, ry0 = int(x0 * sx), int(cy0 * sy)
        cw, ch = int(x1 * sx) - cx0, int(cy1 * sy) - ry0
        g = gs[k].crop(bb)
        c, rad = track(g)
        s = min(cw / g.width, ch / g.height)
        nw, nh = int(g.width * s), int(g.height * s)
        ox, oy = cx0 + (cw - nw) // 2, ry0 + (ch - nh) // 2
        strip.paste(g.resize((nw, nh), Image.LANCZOS), (ox, oy))
        px, py = ox + c[0] * s, oy + c[1] * s
        marks.append((STRIP[0] + px * STRIP[2] / SW, STRIP[1] + py * STRIP[3] / SH,
                      max(rad * s * 2.2, 30) * STRIP[2] / SW))
    b = io.BytesIO(); strip.save(b, 'JPEG', quality=93, optimize=True)
    return base64.b64encode(b.getvalue()).decode(), marks


def jpeg(im, q=93):
    b = io.BytesIO(); im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True)
    return base64.b64encode(b.getvalue()).decode()


def png(im):
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return base64.b64encode(b.getvalue()).decode()


IMGSTY = ('shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;'
          'verticalAlign=top;imageAspect=0;image=data:image/{t},{b};')
INSETSTY = IMGSTY + 'imageBorder=none;strokeWidth=1;imageBackground=none;'
TXTSTY = ('text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={a};'
          'verticalAlign=middle;rounded=0;fontSize=%d;fontFamily={f};fontColor=#000000;' % FS)
ELLSTY = ('ellipse;whiteSpace=wrap;html=1;shapeInside=1;fillColor=none;strokeWidth=8;'
          'strokeColor=light-dark({c},{c});')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--view', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--arms', default='both', choices=['both', 'ours'])
    A = ap.parse_args()
    idxs = [int(x) for x in A.frames.split(',')]
    arms = ['ours', 'frozen'] if A.arms == 'both' else ['ours']

    strip_b64, marks = build_strip(A.obj, idxs)
    pair = {a: [half(A.obj, A.view, i, a) for i in idxs] for a in ('ours', 'frozen')}
    bb = box_union(pair['ours'] + pair['frozen'], 1.0, pad=0.02)
    cells = {a: [x.crop(bb) for x in pair[a]] for a in arms}

    out, n = [], [0]
    def cxml(x, y, w, h, style, value=''):
        n[0] += 1
        # keep &amp;nbsp; escaped: &nbsp; is not a defined XML entity and makes the file
        # unparseable. draw.io renders the escaped form as a non-breaking space anyway,
        # which is exactly what the source figure stores.
        v = SU.escape(value, {'"': '&quot;', "'": '&apos;'})
        out.append(f'<mxCell id="q{n[0]}" value="{v}" style="{style}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    cxml(*TITLE, TXTSTY.format(a='left', f='Comic Sans MS'), A.title)
    cxml(*STRIP, IMGSTY.format(t='jpeg', b=strip_b64))
    for mx, my, mr in marks:
        cxml(mx - mr, my - mr, mr * 2, mr * 2, ELLSTY.format(c=BLOCK['ours']['ring']))
    for arm in arms:
        B = BLOCK[arm]
        cxml(B['label'][0], LABEL_Y, B['label'][1], LABEL_H,
             TXTSTY.format(a='center', f='Helvetica'), ARM_LABEL[arm])
        for i, x in enumerate(B['renders']):
            cxml(x, RENDER_Y, CELL_W, CELL_W, IMGSTY.format(t='jpeg', b=jpeg(cells[arm][i])))
        for i, x in enumerate(B['insets']):
            c, rad = track(cells[arm][i])
            ins = circle_inset(cells[arm][i], c, rad, B['ring'])
            cxml(x, INSET_Y, INSET, INSET, INSETSTY.format(t='png', b=png(ins)))
    if len(arms) > 1:
        x1, y1, x2, y2 = DIVIDER
        n[0] += 1
        out.append(f'<mxCell id="q{n[0]}" style="endArrow=none;dashed=1;html=1;strokeWidth=3;'
                   f'strokeColor=#000000;" edge="1" parent="1"><mxGeometry relative="1" '
                   f'as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                   f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

    xml = (f'<mxfile host="app.diagrams.net">\n<diagram name="{A.name}">\n'
           f'<mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" page="1" '
           f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">\n<root>\n'
           f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(out) +
           '\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n')
    o = E / 'out/BOBDAILIES' / f'{A.name}.drawio.xml'
    o.write_text(xml)
    print(f'page {PAGE_W}x{PAGE_H}  cells {CELL_W:.1f}px  inset {INSET:.0f}px  arms={arms}')
    print(f'  {o}  {o.stat().st_size/1048576:.2f} MB')


if __name__ == '__main__':
    main()
