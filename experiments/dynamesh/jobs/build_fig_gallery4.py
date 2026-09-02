"""build_fig_gallery4.py -- the supplementary gallery: four dynamesh_comparison panels.

Layout constants come from build_fig_drawio (the comparison/failure figure's measured
filmstrip and cells, the title band, the card size that fills the panel width). What this
file does NOT take from there is the cropping, for three reasons that were all visible in
the first draft and are all fixed here:

1. ASPECT. box_union() clamps hw and hh independently:
       hw = min(hw, W/2); hh = min(hh, H/2)
   so a subject tall enough to hit H/2 keeps its width and loses height -- the crop is
   then wider than the cell it is resized into and the subject is SQUEEZED horizontally
   (the statue read "too thin"); a wide subject hits W/2 first and comes out STRETCHED
   (the unicorn read "too wide"). Here the clamp scales both axes by one factor, so the
   crop can only get smaller, never a different shape. The shared module is left alone
   because the duck and flicker figures are already approved against its output.

2. ONE CROP FOR BOTH CAMERAS. collect() computes a separate square_union per camera, so
   the same object comes out at two different sizes in the two rows. Here the union is
   taken over both cameras at once, which is what makes the rows match.

3. CARD CENTRES. XS (cards, computed to fill the width) and XS_GT (film cells, measured
   off the strip) do not agree: card centres land 17px left of the cell above them, so
   every result reads as shifted left of its own video frame. The cards are re-centred on
   the film cells here, and the strip/row gap is asserted rather than assumed.

Per-panel spec, so frame choice and camera pair are per object:
    obj:arm:view_a:view_b:frames:gtpad:prompt
"""
import argparse, base64, sys, xml.sax.saxutils as SU
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv = list(sys.argv); sys.argv = [sys.argv[0], '--obj', 'x', '--view-up', 'x',
                                    '--view-tilt', 'x', '--frames', '1', '--title', 'x',
                                    '--name', 'x']
import build_fig_drawio as B
sys.argv = _argv

E, T2, LAB = B.E, B.T2, B.LAB
GUT_X, GUT_Y = 70.0, 56.0
CELL_CX = [194.7, 519.6, 845.1, 1170.0]                 # film cell centres, measured
XS = [c - B.CARD / 2 for c in CELL_CX]                   # cards centred UNDER their cell


def half(obj, arm, view, idx, side):
    p = E / f'out/view_{obj}_{arm}_{view}/frames/{idx:04d}.png'
    im = Image.open(p).convert('RGB'); w, h = im.size
    s = h - LAB
    return im.crop((0, LAB, s, LAB + s) if side == 'frozen' else (s, LAB, 2 * s, LAB + s))


def fit_union(ims, aspect, pad):
    """Every frame, placed on a shared canvas of exactly `aspect`, never clipped.

    CROPPING CANNOT ALWAYS WORK. The unicorn's content is 635x711 in a 960^2 frame, so a
    1.456:1 box tall enough to hold it would need to be 1035 wide -- wider than the frame
    exists. B.box_union answers that by shrinking height and keeping width, which is the
    squeeze the statue showed; the strict version answers by refusing. Both are wrong: the
    box is not a crop, it is a FRAME, and a frame can be larger than the picture. So the
    canvas is padded with the clip's own background instead, which is invisible on a white
    plate and leaves the subject at its true proportions.

    One shared canvas and one shared content centre for the whole list, so the subject
    neither rescales nor drifts between timesteps.
    """
    bs = [b for b in (B.content_box(i) for i in ims) if b]
    l = min(b[0] for b in bs); t = min(b[1] for b in bs)
    r = max(b[2] for b in bs); d = max(b[3] for b in bs)
    cx, cy = (l + r) / 2, (t + d) / 2
    hw = max((r - l) / 2, (d - t) / 2 * aspect) * (1 + pad)
    hh = hw / aspect
    W = int(round(2 * hw)); H = int(round(2 * hh))
    out = []
    for im in ims:
        px = im.load()
        bg = tuple(int(sum(c[k] for c in (px[0, 0], px[im.size[0] - 1, 0],
                                          px[0, im.size[1] - 1],
                                          px[im.size[0] - 1, im.size[1] - 1])) / 4)
                   for k in range(3))
        canvas = Image.new('RGB', (W, H), bg)
        canvas.paste(im, (int(round(W / 2 - cx)), int(round(H / 2 - cy))))
        out.append(canvas)
    return out


def collect(obj, arm, views, idxs, gt_pad):
    r = {}
    g = [Image.open(T2 / f'data/{obj}/frames_from_video/frame_{i:04d}.png').convert('RGB')
         for i in idxs]
    r['gt'] = [B.jpeg(x.resize((int(B.GT_W * 2), int(B.GT_H * 2)), Image.LANCZOS))
               for x in fit_union(g, B.GT_W / B.GT_H, gt_pad)]
    # ONE canvas shared by BOTH cameras: a per-camera crop draws the same object at two
    # different sizes, which is exactly the "second view too big" note.
    per = {v: [half(obj, arm, v, i, 'ours') for i in idxs] for v in views}
    both = fit_union(per[views[0]] + per[views[1]], 1.0, 0.03)
    k = len(idxs)
    for tag, sl in (('up', slice(0, k)), ('tilt', slice(k, 2 * k))):
        r[tag] = [B.jpeg(x.resize((620, 620), Image.LANCZOS)) for x in both[sl]]
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', nargs='+', required=True,
                    help='obj:arm:view_a:view_b:frames:gtpad:prompt (four, row-major)')
    ap.add_argument('--name', default='dynamesh_gallery4')
    A = ap.parse_args()
    assert len(A.spec) == 4, 'this page is a 2x2 grid; give exactly four'

    L, PH = B.layout(['ours'])
    # GATE-overlap: the first result row must start BELOW the filmstrip. The goat's
    # cards touching the black frame is the failure this catches.
    sx, sy, sw, sh = L['strip']
    assert L['ours_up_y'] >= sy + sh, \
        f"row 1 at y={L['ours_up_y']} overlaps the strip ending at {sy + sh}"
    strip = base64.b64encode((E / 'out/BOBDAILIES/filmstrip_border.jpeg').read_bytes()).decode()
    PW = B.PAGE_W

    cells, n, panels = [], [0], []
    def emit(x, y, w, h, style, value=''):
        n[0] += 1
        v = SU.escape(value, {'"': '&quot;', "'": '&apos;'})
        cells.append(f'<mxCell id="g{n[0]}" value="{v}" style="{style}" vertex="1" '
                     f'parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                     f'height="{h:.1f}" as="geometry"/></mxCell>')

    for k, spec in enumerate(A.spec):
        obj, arm, v_a, v_b, frames, gtpad, prompt = spec.split(':', 6)
        idxs = [int(x) for x in frames.split(',')]
        dx, dy = (k % 2) * (PW + GUT_X), (k // 2) * (PH + GUT_Y)
        print(f'  panel {k}: {obj} [{arm}] {v_a}/{v_b} frames {idxs} gtpad {gtpad}', flush=True)
        img = collect(obj, arm, (v_a, v_b), idxs, float(gtpad))
        panels.append((img, prompt, dx, dy))
        x, y, w, h = L['title']
        emit(x + dx, y + dy, w, h, B.TXTSTY.format(a='left', f=B.FONT_TITLE), f'"{prompt}"')
        emit(sx + dx, sy + dy, sw, sh, B.IMGSTY.format(b=strip, rot=''))
        for i, gx in enumerate(B.XS_GT):
            emit(gx + dx, L['gt_y'] + dy, B.GT_W, B.GT_H, B.IMGSTY.format(b=img['gt'][i], rot=''))
        for tag in ('up', 'tilt'):
            for i, cx in enumerate(XS):
                emit(cx + dx, L[f'ours_{tag}_y'] + dy, B.CARD, B.CARD,
                     B.IMGSTY.format(b=img[tag][i], rot=''))

    W, H = 2 * PW + GUT_X, 2 * PH + GUT_Y
    xml = (f'<mxfile host="app.diagrams.net">\n<diagram name="{A.name}">\n'
           f'<mxGraphModel dx="1000" dy="700" grid="0" gridSize="10" page="1" '
           f'pageWidth="{W:.0f}" pageHeight="{H:.0f}" math="0" shadow="0">\n<root>\n'
           f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n' + '\n'.join(cells) +
           '\n</root>\n</mxGraphModel>\n</diagram>\n</mxfile>\n')

    def im(x, y, w, h, b):
        return (f'<img src="data:image/jpeg;base64,{b}" style="left:{x:.1f}px;top:{y:.1f}px;'
                f'width:{w:.1f}px;height:{h:.1f}px;">')
    P = []
    for img, prompt, dx, dy in panels:
        x, y, w, h = L['title']
        P.append(f'<div class="t" style="left:{x+dx:.1f}px;top:{y+dy:.1f}px;width:{w:.1f}px;'
                 f'height:{h:.1f}px;font-family:\'{B.FONT_TITLE}\',cursive,sans-serif;'
                 f'text-align:left">{SU.escape(chr(34)+prompt+chr(34))}</div>')
        P.append(im(sx + dx, sy + dy, sw, sh, strip))
        P += [im(gx + dx, L['gt_y'] + dy, B.GT_W, B.GT_H, img['gt'][i])
              for i, gx in enumerate(B.XS_GT)]
        for tag in ('up', 'tilt'):
            P += [im(cx + dx, L[f'ours_{tag}_y'] + dy, B.CARD, B.CARD, img[tag][i])
                  for i, cx in enumerate(XS)]
    page = '\n'.join(P)
    html = f"""<title>DynaMesh Gallery Panel</title>
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
.page{{position:relative;width:{W:.0f}px;height:{H:.0f}px;background:#fff;transform-origin:top left}}
.page img{{position:absolute;object-fit:contain}}
.page .t{{position:absolute;font-size:{B.FS}px;color:#000;line-height:1.08;white-space:nowrap}}
.legend{{margin-top:20px;display:flex;flex-direction:column;gap:8px;font-size:14px;color:var(--text-dim)}}
.legend b{{color:var(--text)}} code{{font-family:var(--mono);font-size:12.5px}}
</style>
<header><div class="in">
  <span class="eyebrow">3DV &apos;27 &middot; DynaMesh &middot; supplementary gallery</span>
  <h1>Four objects, two cameras each</h1>
  <p class="sub">Row 1: unicorn, goat. Row 2: fish, statue. Each panel is the driving clip
  in the filmstrip, then the result at two cameras 180&deg; apart, at the same four timesteps.</p>
</div></header>
<main>
  <div class="scroll"><div class="page" id="pg">{page}</div></div>
  <div class="legend">
    <div><b>Cameras</b> &mdash; the second row is the first rotated 180&deg;, so the back of the object is shown.</div>
    <div><b>One crop per object</b>, shared by both cameras, so the two rows are the same size.</div>
    <div>Page <code>{W:.0f}&times;{H:.0f}</code>, cards <code>{B.CARD:.0f}px</code> centred on the film cells above them.</div>
  </div>
</main>
<script>
function fit(){{const p=document.getElementById('pg'),w=p.parentElement.clientWidth;
  const s=Math.min(1,w/{W:.0f});p.style.transform='scale('+s+')';
  p.parentElement.style.height=({H:.0f}*s)+'px';}}
fit();addEventListener('resize',fit);
</script>"""

    canvas = Image.new('RGB', (int(W), int(H)), 'white')
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', B.FS)
    except Exception:
        font = ImageFont.load_default()
    def paste(b, x, y, w, h):
        canvas.paste(Image.open(BytesIO(base64.b64decode(b))).convert('RGB')
                     .resize((int(w), int(h)), Image.LANCZOS), (int(x), int(y)))
    for img, prompt, dx, dy in panels:
        tx_, ty_, _, _ = L['title']
        dr.text((tx_ + dx, ty_ + dy), f'"{prompt}"', fill='black', font=font)
        paste(strip, sx + dx, sy + dy, sw, sh)
        for i, gx in enumerate(B.XS_GT):
            paste(img['gt'][i], gx + dx, L['gt_y'] + dy, B.GT_W, B.GT_H)
        for tag in ('up', 'tilt'):
            for i, cx in enumerate(XS):
                paste(img[tag][i], cx + dx, L[f'ours_{tag}_y'] + dy, B.CARD, B.CARD)

    o = E / 'out/BOBDAILIES'
    (o / f'{A.name}.drawio.xml').write_text(xml)
    (o / f'{A.name}.html').write_text(html)
    canvas.save(o / f'{A.name}.png')
    print(f'page {W:.0f}x{H:.0f}  cards {B.CARD:.1f}px centred on {CELL_CX}')
    print(f'strip ends y={sy+sh:.0f}, row 1 starts y={L["ours_up_y"]:.0f} -- no overlap')
    for f in (f'{A.name}.drawio.xml', f'{A.name}.html', f'{A.name}.png'):
        print(f'  {o/f}  {(o/f).stat().st_size/1048576:.2f} MB')
