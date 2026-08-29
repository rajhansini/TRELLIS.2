"""build_fig_rows_small.py -- the ORIGINAL two-row duck figure, at flicker_trimmed's size.

Layout is the one we already had: quoted prompt, driving-video strip, then two rows of four
renders (two novel cameras), ours only, no method headline, no insets, no divider.

Only the SCALE changes: page 1243 wide and ~546 tall, title Comic Sans 44, strip 588.6x151.7
centred -- every number lifted from dynamesh_flicker_trimmed.drawio rather than invented.
Cells are landscape because that is what fits two rows inside 546px at this width, and a
wide duck fills a landscape cell better than a square one anyway.
"""
import argparse, base64, io, sys, xml.sax.saxutils as SU
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fig_trimmed import (E, T2, half, box_union, track, build_strip, jpeg,
                               IMGSTY, TXTSTY, ELLSTY)

PAGE_W = 1243
MARGIN, GAP = 20.0, 14.0
CELL_W = (PAGE_W - 2 * MARGIN - 3 * GAP) / 4          # 290.25
CELL_H = 160.0
TITLE = (MARGIN, 0.0, PAGE_W - 2 * MARGIN, 52.0)
STRIP = (327.2, 58.0, 588.6, 151.7)                    # centred, exactly the source's box
FS = 44


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--view', required=True)
    ap.add_argument('--view2', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--arm', default='ours', choices=['ours', 'frozen'])
    ap.add_argument('--name', required=True)
    A = ap.parse_args()
    idxs = [int(x) for x in A.frames.split(',')]

    import build_fig_trimmed as BT
    BT.STRIP = STRIP
    strip_b64, _ = build_strip(A.obj, idxs)

    def row(view):
        ims = [half(A.obj, view, i, A.arm) for i in idxs]
        bb = box_union(ims, CELL_W / CELL_H, pad=0.02)
        return [x.crop(bb) for x in ims]

    r1, r2 = row(A.view), row(A.view2)
    y1 = STRIP[1] + STRIP[3] + 14
    y2 = y1 + CELL_H + 10
    page_h = int(y2 + CELL_H + 14)
    xs = [MARGIN + i * (CELL_W + GAP) for i in range(4)]

    out, n = [], [0]
    def cxml(x, y, w, h, style, value=''):
        n[0] += 1
        v = SU.escape(value, {'"': '&quot;', "'": '&apos;'})
        out.append(f'<mxCell id="r{n[0]}" value="{v}" style="{style}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    cxml(*TITLE, TXTSTY.format(a='left', f='Comic Sans MS'), A.title)
    cxml(*STRIP, IMGSTY.format(t='jpeg', b=strip_b64))
    for cells, y in ((r1, y1), (r2, y2)):
        for i, x in enumerate(xs):
            cxml(x, y, CELL_W, CELL_H, IMGSTY.format(t='jpeg', b=jpeg(cells[i])))

    xml = (f'<mxfile host="app.diagrams.net">\n<diagram name="{A.name}">\n'
           f'<mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" page="1" '
           f'pageWidth="{PAGE_W}" pageHeight="{page_h}" math="0" shadow="0">\n<root>\n'
           f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(out) +
           '\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n')
    o = E / 'out/BOBDAILIES' / f'{A.name}.drawio.xml'
    o.write_text(xml)
    print(f'page {PAGE_W}x{page_h}  cells {CELL_W:.1f}x{CELL_H:.0f}  title fs{FS}')
    print(f'  {o}  {o.stat().st_size/1048576:.2f} MB')


if __name__ == '__main__':
    main()
