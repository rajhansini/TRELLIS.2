"""build_daily_page.py -- the per-object dailies page, matching the batch-C artifacts.

Four view tabs (train + three unseen diagonals), each a four-column mp4 inlined as a
data URI. Same markup, same CSS tokens and same selectView() as the existing pages, so a
continuation object sits in the gallery beside the ladder objects without looking foreign.
"""
import base64, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
VIEWS = [('train', 'Training view'), ('diagA', 'Diagonal A'),
         ('diagB', 'Diagonal B'), ('diagC', 'Diagonal C')]

CSS = """
:root{
  --bg:#f7f5f1; --surface:#ffffff; --surface-2:#eeece6; --border:#ddd9d0;
  --text:#1c1d22; --text-dim:#5c5f6b; --text-faint:#8b8d98;
  --accent:#b9762c; --accent-ink:#ffffff; --accent-soft:#f1e2ce;
  --shadow: 0 1px 2px rgba(20,18,14,.06), 0 8px 24px rgba(20,18,14,.06);
  --radius: 10px;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
  --sans: 'IBM Plex Sans', -apple-system, sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#15161b; --surface:#1c1e25; --surface-2:#262933; --border:#34384a;
    --text:#eceef2; --text-dim:#9195a3; --text-faint:#6b6f7d;
    --accent:#d9974f; --accent-ink:#1a1200; --accent-soft:#3a2c17;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 28px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --bg:#15161b; --surface:#1c1e25; --surface-2:#262933; --border:#34384a;
  --text:#eceef2; --text-dim:#9195a3; --text-faint:#6b6f7d;
  --accent:#d9974f; --accent-ink:#1a1200; --accent-soft:#3a2c17;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 28px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0; background:var(--bg); color:var(--text); font-family:var(--sans);
  -webkit-font-smoothing:antialiased; line-height:1.5;}
a{color:inherit}
.wrap{max-width:1040px; margin:0 auto; padding:0 28px 96px;}
header.top{padding:40px 28px 28px; border-bottom:1px solid var(--border);
  display:flex; flex-direction:column; gap:6px; background:var(--surface);}
header.top .inner{max-width:1040px; margin:0 auto; width:100%; display:flex; flex-direction:column; gap:8px;}
.eyebrow{font-family:var(--mono); font-size:12px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--accent); font-weight:600;}
h1{font-size:30px; font-weight:700; margin:0; letter-spacing:-.01em; text-wrap:balance;}
.meta{font-family:var(--mono); font-size:12.5px; color:var(--text-dim);}
.backlink{font-family:var(--mono); font-size:12.5px; color:var(--text-dim); text-decoration:none;
  display:inline-flex; align-items:center; gap:6px; margin-top:4px;}
.backlink:hover{color:var(--accent)}
.tabbar{display:flex; gap:6px; flex-wrap:wrap; margin:22px 0 4px;}
.tab{font-family:var(--mono); font-size:13px; padding:9px 16px; border-radius:999px;
  border:1px solid var(--border); background:var(--surface); color:var(--text-dim);
  cursor:pointer; transition:background .15s,color .15s,border-color .15s;}
.tab:hover{color:var(--text); border-color:var(--text-faint)}
.tab.active{background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:600;}
.camline{font-family:var(--mono); font-size:12.5px; color:var(--text-faint); margin:2px 0 18px; min-height:1.2em;}
.stage{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow); overflow:hidden;}
.panel-video{width:100%; display:none; background:#16161a;}
.panel-video.active{display:block;}
.legend{margin-top:22px; padding:18px 20px; background:var(--surface-2); border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:8px; font-size:13.5px; color:var(--text-dim);
  display:flex; flex-direction:column; gap:9px;}
.legend b{color:var(--text);}
.mono{font-family:var(--mono); font-size:.92em;}
table.num{width:100%; border-collapse:collapse; font-size:13.5px; margin:22px 0 0;}
table.num th{text-align:left; font-family:var(--mono); font-size:11px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--text-faint); font-weight:600;
  padding:6px 10px 6px 0; border-bottom:1px solid var(--border);}
table.num td{padding:8px 10px 8px 0; border-bottom:1px solid var(--border); font-variant-numeric:tabular-nums;}
table.num tr.hi td{background:var(--accent-soft);}
.objnav{display:flex; gap:8px; flex-wrap:wrap; margin-top:26px;}
.pill{font-family:var(--mono); font-size:12.5px; padding:7px 13px; border-radius:999px;
  border:1px solid var(--border); background:var(--surface); color:var(--text-dim); text-decoration:none;}
.pill:hover{border-color:var(--accent); color:var(--text)}
.pill.current{background:var(--accent-soft); border-color:var(--accent); color:var(--text); font-weight:600;}
"""

CAM = ("{'train':'yaw 0\\u00b0  \\u00b7  elev 0\\u00b0  \\u00b7  supervised',"
       "'diagA':'yaw 45\\u00b0  \\u00b7  elev +25\\u00b0  \\u00b7  unseen',"
       "'diagB':'yaw 135\\u00b0  \\u00b7  elev \\u221220\\u00b0  \\u00b7  unseen',"
       "'diagC':'yaw 225\\u00b0  \\u00b7  elev +30\\u00b0  \\u00b7  unseen'}")


def enc(p):
    return 'data:video/mp4;base64,' + base64.b64encode(Path(p).read_bytes()).decode()


def build(obj, title, mesh, effect, gallery_url, extra_rows='', legend_extra='',
          cols_line=None, eyebrow='Continuation Dailies &middot; round 2', out=None,
          origin_line=None):
    # The column legend is a PARAMETER because not every object has four columns.
    # Objects trained on the MCFM arm only ship a three-column daily, and a page
    # that says 'four columns' over a three-column video is worse than no legend.
    # origin_line is a PARAMETER for the same reason cols_line is. The continuation
    # objects were handed our own round-1 render at frame 150, so their texture is
    # already there at frame 1; batch G's clips start from the grey mesh and the
    # texture ARRIVES. Printing the wrong one tells the reader to look for the wrong
    # thing in the first 20 frames.
    origin_line = origin_line or ('<b>This object did not start from a grey mesh.</b> '
        'Kling was handed our own round-1 fitted render at frame 150 and asked to keep '
        'the texture moving. The driving video is texture in motion, not texture arriving.')
    cols_line = cols_line or ('<b>Four columns</b> &mdash; ground truth, frozen TRELLIS.2, rung27, rung27 + MCFM temporal-only (3-frame window, <span class="mono">v2_D</span>).')
    vids = ''.join(
        f'<video class="panel-video{" active" if i==0 else ""}" id="v-{v}" '
        f'src="{enc(E/"out/DAILY"/f"{obj}_{v}.mp4")}" controls muted loop playsinline '
        f'preload="{"auto" if i==0 else "none"}"></video>'
        for i, (v, _) in enumerate(VIEWS))
    tabs = ''.join(
        f'<button class="tab{" active" if i==0 else ""}" data-view="{v}" '
        f'onclick="selectView(\'{v}\')">{lab}</button>'
        for i, (v, lab) in enumerate(VIEWS))
    html = f"""<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<header class="top"><div class="inner">
  <a class="backlink" href="{gallery_url}">&larr; all objects</a>
  <span class="eyebrow">{eyebrow}</span>
  <h1>{title}</h1>
  <div class="meta">mesh: {mesh}  &middot;  effect: {effect}</div>
</div></header>
<main class="wrap">
  <div class="tabbar">{tabs}</div>
  <div class="camline" id="camline"></div>
  <div class="stage">{vids}</div>
  <div class="legend">
    <div>{cols_line}</div>
    <div>{origin_line}</div>
    <div><b>Camera</b> &mdash; fixed per clip; every frame-to-frame change is texture, never camera motion.</div>
    <div><b>Ground truth</b> away from the training view shows the driving-video frame dimmed &mdash; there is no ground truth from an angle the camera never shot.</div>
    {legend_extra}
  </div>
  {extra_rows}
</main>
<script>
const CAM = {CAM};
function selectView(v){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.view===v));
  document.querySelectorAll('.panel-video').forEach(el=>{{
    const on = el.id === 'v-'+v;
    el.classList.toggle('active', on);
    if(on && el.preload!=='auto'){{ el.preload='auto'; }}
    if(!on) el.pause();
  }});
  document.getElementById('camline').textContent = CAM[v] || '';
}}
selectView('train');
</script>"""
    out = Path(out or f'/tmp/{obj}_daily.html')
    out.write_text(html)
    print(f'{out}  {out.stat().st_size/1024/1024:.2f} MB')
    return out
