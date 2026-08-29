"""build_camera_options.py -- one page showing every rendered camera for an object, so a
camera can be CHOSEN by looking rather than by argument.

Each row is one camera at the four figure frames, so the angle and the dot motion are
judged together -- a camera that reads well on a still can still be the wrong one if the
dots barely move on it.

Two numbers per row, both measured on the actual renders, not predicted:
  unpainted%  near-white pixels inside the silhouette -- surface the training camera
              never saw and the adapter therefore never learned to paint
  motion      mean per-frame change across the clip, as a proxy for visible drift
"""
import base64, io, sys
from pathlib import Path
import numpy as np
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OBJ = sys.argv[1] if len(sys.argv) > 1 else 'bob_spots'
FRAMES = [50, 75, 100, 125]
VIEWS = [
    ('diagB', 135, -20), ('sideH', 210, 0), ('sideG', 230, 0), ('sideI', 230, 5),
    ('topE', 180, 12), ('topC', 180, 25), ('diagC', 225, 30), ('topD', 180, 35),
    ('topA', 180, 50), ('topF', 180, 75), ('train', 0, 0), ('diagA', 45, 25),
    ('upA', 230, 10), ('upB', 240, 10), ('upC', 230, 15),
    ('eyeD', 60, 5), ('eyeE', 30, 15), ('eyeF', 75, 10), ('eyeG', 90, 0), ('eyeH', 105, 0),
    ('d01', 50, -8), ('d02', 50, 0), ('d03', 50, 8), ('d04', 50, 16), ('d05', 60, -8), ('d06', 60, 0), ('d07', 60, 8), ('d08', 60, 16), ('d09', 70, -8), ('d10', 70, 0), ('d11', 70, 8), ('d12', 70, 16),
]
EYE_SIDE = {'train','diagA','eyeD','eyeE','eyeF','eyeG','eyeH'} | {f'd{i:02d}' for i in range(1,13)}      # the malformed eye is only visible near yaw 0


def half(view, idx):
    im = Image.open(E / f'out/view_{OBJ}_27m_{view}/frames/{idx:04d}.png').convert('RGB')
    w, h = im.size; s = h - 28
    return im.crop((s, 28, 2 * s, 28 + s))


def stats(view):
    ups, prev, mot = [], None, []
    for i in FRAMES:
        a = np.asarray(half(view, i)).astype(np.int16)
        lum = a.mean(-1); sat = a.max(-1) - a.min(-1); body = lum < 250
        ups.append(100 * ((lum > 205) & (sat < 38) & body).sum() / body.sum())
        sm = np.asarray(half(view, i).resize((120, 120))).astype(np.float32)
        if prev is not None:
            mot.append(np.abs(sm - prev).mean())
        prev = sm
    return float(np.mean(ups)), float(np.mean(mot))


def b64(im, px=300, q=86):
    bb = io.BytesIO()
    im.resize((px, px), Image.LANCZOS).save(bb, 'JPEG', quality=q, optimize=True)
    return base64.b64encode(bb.getvalue()).decode()


rows = []
for v, yaw, elev in VIEWS:
    d = E / f'out/view_{OBJ}_27m_{v}/frames'
    if not d.is_dir() or len(list(d.glob('*.png'))) < max(FRAMES):
        continue
    up, mo = stats(v)
    tone = 'good' if up < 0.1 else 'warn' if up < 0.6 else 'bad'
    eye = ('<span class="chip warn">eye side</span>' if v in EYE_SIDE
           else '<span class="chip good">no eye</span>')
    tilt = ('below' if elev < 0 else 'level' if elev == 0 else
            'slightly above' if elev <= 15 else 'above' if elev <= 35 else 'overhead')
    imgs = ''.join(f'<img src="data:image/jpeg;base64,{b64(half(v, i))}" alt="{v} frame {i}">'
                   for i in FRAMES)
    rows.append(f'''<section class="row">
  <div class="meta">
    <p class="nm">{v}</p>
    <p class="cam">yaw {yaw}&deg; &middot; elev {elev:+d}&deg;</p>
    <p class="tilt">{tilt}</p>
    <p class="stat {tone}">unpainted {up:.2f}%</p>
    <p class="stat">motion {mo:.1f}</p>
    {eye}
  </div>
  <div class="strip">{imgs}</div>
</section>''')

HTML = f"""<title>Duck Camera Options</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{{--bg:#f7f5f1;--surface:#fff;--surface-2:#eeece6;--border:#ddd9d0;--text:#1c1d22;--text-dim:#5c5f6b;--text-faint:#8b8d98;--accent:#b9762c;--good:#2f7d54;--warn:#a4701b;--bad:#a33a33;--mono:'IBM Plex Mono',ui-monospace,monospace;--sans:'IBM Plex Sans',-apple-system,sans-serif;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#15161b;--surface:#1c1e25;--surface-2:#262933;--border:#34384a;--text:#eceef2;--text-dim:#9195a3;--text-faint:#6b6f7d;--accent:#d9974f;--good:#63c48c;--warn:#dcae5c;--bad:#e08079;}}}}
:root[data-theme="dark"]{{--bg:#15161b;--surface:#1c1e25;--surface-2:#262933;--border:#34384a;--text:#eceef2;--text-dim:#9195a3;--text-faint:#6b6f7d;--accent:#d9974f;--good:#63c48c;--warn:#dcae5c;--bad:#e08079;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}}
header{{padding:38px 28px 24px;border-bottom:1px solid var(--border);background:var(--surface)}}
header .in{{max-width:1120px;margin:0 auto;display:flex;flex-direction:column;gap:7px}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:600}}
h1{{font-size:29px;margin:0;letter-spacing:-.01em}}
.sub{{color:var(--text-dim);font-size:15px;margin:0;max-width:72ch}}
main{{max-width:1120px;margin:0 auto;padding:26px 28px 90px;display:flex;flex-direction:column;gap:12px}}
.row{{display:grid;grid-template-columns:184px 1fr;gap:16px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;align-items:center}}
.meta{{display:flex;flex-direction:column;gap:2px}}
.nm{{font-family:var(--mono);font-size:16px;font-weight:600;margin:0;color:var(--text)}}
.cam{{font-family:var(--mono);font-size:12.5px;color:var(--text-dim);margin:0;font-variant-numeric:tabular-nums}}
.tilt{{font-size:12.5px;color:var(--text-faint);margin:0 0 4px}}
.stat{{font-family:var(--mono);font-size:12.5px;margin:0;color:var(--text-dim);font-variant-numeric:tabular-nums}}
.stat.good{{color:var(--good);font-weight:600}} .stat.warn{{color:var(--warn);font-weight:600}} .stat.bad{{color:var(--bad);font-weight:600}}
.chip{{display:inline-block;margin-top:6px;font-family:var(--mono);font-size:10.5px;padding:3px 8px;border-radius:999px;border:1px solid var(--border);background:var(--surface-2);color:var(--text-dim);width:fit-content}}
.chip.good{{color:var(--good);border-color:var(--good)}} .chip.bad{{color:var(--bad);border-color:var(--bad)}} .chip.warn{{color:var(--warn);border-color:var(--warn)}}
.strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;min-width:0}}
.strip img{{width:100%;height:auto;display:block;background:#fff;border-radius:6px}}
.note{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px;font-size:14px;color:var(--text-dim);line-height:1.65}}
.note b{{color:var(--text)}} code{{font-family:var(--mono);font-size:12.5px}}
</style>
<header><div class="in">
  <span class="eyebrow">3DV &apos;27 &middot; DynaMesh &middot; camera pick</span>
  <h1>Duck Camera Options</h1>
  <p class="sub">Every rendered camera for <code>{OBJ}</code>, each at the four figure frames
  (50 / 75 / 100 / 125). Sorted cleanest first. Name two and I&apos;ll rebuild the figure.</p>
</div></header>
<main>
{''.join(rows)}
  <div class="note">
    <b>Why elevation and the eye pull against each other.</b> The malformed eye sits near
    yaw 0, so an eye-free camera must be on the far hemisphere &mdash; and on that
    hemisphere every degree of elevation opens the ring&apos;s inner basin, which the
    training camera never saw and the adapter therefore left unpainted. That is why the
    cleanest rows are also the lowest ones.
    <br><br><b>unpainted%</b> is near-white pixels inside the silhouette, averaged over the
    four frames. Below about 0.1% nothing is visible; by 0.4% a pale patch reads clearly.
  </div>
</main>"""

o = E / 'out/BOBDAILIES/camera_options.html'
o.write_text(HTML)
print(f'{o}  {o.stat().st_size/1048576:.2f} MB  ({len(rows)} cameras)')
