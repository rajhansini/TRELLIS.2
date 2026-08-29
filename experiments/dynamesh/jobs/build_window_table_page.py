#!/usr/bin/env python3
"""build_window_table_page.py -- render out/window_table_all.json as one HTML table."""
import json

E = '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
d = json.load(open(f'{E}/out/window_table_all.json'))
rows = d['rows']
n = d['n_objects']

# best-in-column over every scored row; GT is a reference, not a competitor, and
# drift is a guard (closest to GT is not "best"), so neither is ever marked.
COLS = [('s_flicker', 'lo'), ('s_accel', 'lo'), ('s_drift', None),
        ('u_flicker', 'lo'), ('u_accel', 'lo'), ('u_drift', None),
        ('psnr', 'hi'), ('ssim', 'hi')]
cand = [r for r in rows if r['name'] != 'Ground truth']
best = {}
for k, dirn in COLS:
    if not dirn:
        continue
    vals = [r[k] for r in cand if k in r]
    best[k] = (min if dirn == 'lo' else max)(vals)

def cell(r, k, fmt):
    if k not in r:
        return '<td class="na">&mdash;</td>'
    v = r[k]
    cls = ' class="best"' if k in best and abs(v - best[k]) < 1e-12 else ''
    return f'<td{cls}>{v:{fmt}}</td>'

body = []
for i, r in enumerate(rows):
    nm = r['name']
    cls = []
    if nm == 'Ground truth':
        cls.append('ref')
    if nm.startswith('DynaMesh'):
        cls.append('ours')
    if nm == 'DynaMesh (ours, W=3)':
        cls.append('sep')
    label = nm
    note = ''
    if nm.startswith('DynaMesh'):
        w = nm.split('W=')[1].rstrip(')')
        label = f'DynaMesh (ours), <span class="w">W&thinsp;=&thinsp;{w}</span>'
        if w == '3':
            note = '<span class="tag">published</span>'
    tds = ''.join([cell(r, 's_flicker', '.5f'), cell(r, 's_accel', '.5f'), cell(r, 's_drift', '.4f'),
                   cell(r, 'u_flicker', '.5f'), cell(r, 'u_accel', '.5f'), cell(r, 'u_drift', '.4f'),
                   cell(r, 'psnr', '.2f'), cell(r, 'ssim', '.4f')])
    body.append(f'<tr class="{" ".join(cls)}"><td class="lab">{label}{note}</td>{tds}</tr>')

html = f'''<title>Window Sweep Table</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root{{
  --ground:#eceef2; --surface:#fff; --rule:#d3d9e1; --rule-soft:#e6eaf0;
  --ink:#141821; --dim:#59616e; --faint:#8b93a0;
  --accent:#0f6f68; --ours:#e6f0ee; --ours-rule:#bcd6d1;
  --mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
  --sans:'Public Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --ground:#11141a; --surface:#181c23; --rule:#2d3540; --rule-soft:#232a33;
  --ink:#e7ebf1; --dim:#98a1af; --faint:#6c7583;
  --accent:#54c0b2; --ours:#15302c; --ours-rule:#255049;
}}}}
:root[data-theme="dark"]{{
  --ground:#11141a; --surface:#181c23; --rule:#2d3540; --rule-soft:#232a33;
  --ink:#e7ebf1; --dim:#98a1af; --faint:#6c7583;
  --accent:#54c0b2; --ours:#15302c; --ours-rule:#255049;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1040px;margin:0 auto;padding:56px 24px 72px;display:flex;
  flex-direction:column;gap:22px}}
.head{{display:flex;flex-direction:column;gap:8px}}
.eyebrow{{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent)}}
h1{{font-family:var(--serif);font-size:clamp(28px,4vw,38px);font-weight:600;margin:0;
  letter-spacing:-.012em;text-wrap:balance}}
.cap{{margin:0;color:var(--dim);font-size:14.5px;max-width:66ch}}
.cap b{{color:var(--ink);font-weight:600}}
.card{{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
  overflow-x:auto;box-shadow:0 1px 2px rgba(16,20,28,.05),0 10px 30px rgba(16,20,28,.055)}}
table{{width:100%;min-width:900px;border-collapse:collapse;
  font-variant-numeric:tabular-nums}}
th,td{{text-align:right;white-space:nowrap;padding:11px 12px}}
thead .grp th{{font-family:var(--sans);font-size:11px;font-weight:600;letter-spacing:.09em;
  text-transform:uppercase;color:var(--dim);text-align:center;padding:16px 12px 5px}}
thead .grp th.g{{border-bottom:1px solid var(--rule-soft)}}
thead .col th{{font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:.05em;
  text-transform:uppercase;color:var(--faint);padding:5px 12px 12px;
  border-bottom:1px solid var(--rule)}}
thead th:first-child{{text-align:left}}
tbody td{{font-family:var(--mono);font-size:13px;border-bottom:1px solid var(--rule-soft)}}
tbody td.lab{{text-align:left;font-family:var(--sans);font-size:13.5px;font-weight:500;
  white-space:normal}}
tbody tr:last-child td{{border-bottom:none}}
td.best{{font-weight:700;color:var(--accent)}}
td.na{{color:var(--faint)}}
tr.ref td{{color:var(--faint);font-style:italic}}
tr.ours td{{background:var(--ours)}}
tr.ours td.lab{{font-weight:600}}
tr.sep td{{border-top:1px solid var(--rule)}}
.w{{font-family:var(--mono);font-weight:700;color:var(--accent)}}
.tag{{font-family:var(--mono);font-size:9.5px;font-weight:500;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim);border:1px solid var(--ours-rule);
  border-radius:4px;padding:1px 5px;margin-left:8px;vertical-align:1px}}
.foot{{font-family:var(--mono);font-size:11.5px;line-height:1.85;color:var(--faint);margin:0}}
@media (max-width:640px){{.wrap{{padding:36px 16px 56px}}}}
</style>
<div class="wrap">
  <div class="head">
    <div class="eyebrow">3DV &rsquo;27 &middot; DynaMesh</div>
    <h1>Window sweep against every baseline</h1>
    <p class="cap">All methods and all four MCFM conditioning windows on one reference.
      <b>{n} objects</b>, 150 frames, 512&nbsp;px, per-method foreground. Baseline rows are
      identical across the four window runs; only the DynaMesh row moves. Drift is a guard,
      not a score, so it is never marked best.</p>
  </div>
  <div class="card">
    <table>
      <thead>
        <tr class="grp"><th></th><th class="g" colspan="3">Supervised view</th>
          <th class="g" colspan="3">Unseen views</th><th class="g" colspan="2">Fidelity vs. GT</th></tr>
        <tr class="col"><th>Method</th>
          <th>Flicker&darr;</th><th>Accel.&darr;</th><th>Drift</th>
          <th>Flicker&darr;</th><th>Accel.&darr;</th><th>Drift</th>
          <th>PSNR&uarr;</th><th>SSIM&uarr;</th></tr>
      </thead>
      <tbody>
{chr(10).join("        " + b for b in body)}
      </tbody>
    </table>
  </div>
  <p class="foot">Windows: W=3 out/FULLRATE_CG &middot; W=5 out/FULLRATE_W5 &middot;
    W=7 out/FULLRATE_W7 &middot; W=11 out/FULLRATE_W11CG, all scored by
    jobs/fullrate_metrics_arm.py against the 2D copy out/gt_targets_&lt;obj&gt;/frames.<br>
    Validation gate: every non-DynaMesh cell diffed against the published evaluation,
    0 outside 12%. Baseline spread across the four runs: {d['baseline_max_spread']:.0e}.</p>
</div>
'''

open(f'{E}/out/window_table_page.html', 'w').write(html)
print(f'-> {E}/out/window_table_page.html   ({len(html)} bytes, {len(rows)} rows, {n} objects)')
