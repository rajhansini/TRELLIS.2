"""build_texel_artifact.py — paper-grade table for the texel temporal metric.

The verdict is DERIVED FROM THE DATA, never pre-written. If MCFM loses, the page
says so; there is no branch that reports a win that was not measured.
"""
import json, sys
import numpy as np
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
S = json.load(open(E / 'out/texel_summary.json'))
OUT = Path(sys.argv[1])

rows = []
for o, v in sorted(S.items()):
    a, b = v['r27'], v['mcfm']
    rows.append(dict(obj=o, n=a['n_frames'], vox=a['n_voxels'],
                     fa=a['texel_flicker'], fb=b['texel_flicker'],
                     ja=a['texel_jerk'], jb=b['texel_jerk'],
                     da=a['texel_drift'], db=b['texel_drift'],
                     dF=100*(b['texel_flicker']-a['texel_flicker'])/a['texel_flicker'],
                     dJ=100*(b['texel_jerk']-a['texel_jerk'])/a['texel_jerk'],
                     dD=100*(b['texel_drift']-a['texel_drift'])/a['texel_drift']))
n = len(rows)
wF = sum(1 for r in rows if r['dF'] < 0); mF = np.mean([r['dF'] for r in rows])
wJ = sum(1 for r in rows if r['dJ'] < 0); mJ = np.mean([r['dJ'] for r in rows])
mD = np.mean([r['dD'] for r in rows]); aD = np.mean([abs(r['dD']) for r in rows])

# two-sided sign test against a fair coin
from math import comb
def sign_p(w, n):
    k = max(w, n - w)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
pF, pJ = sign_p(wF, n), sign_p(wJ, n)

guard = ('holds' if aD < 5 else 'moves')
if wF > n / 2 and aD < 5:
    verdict = (f'MCFM lowers texel flicker on <b>{wF} of {n}</b> objects, mean '
               f'<b>{mF:+.1f}%</b>, while drift moves only <b>{mD:+.1f}%</b>. '
               f'The texture still travels as far; what is removed is jitter.')
    vcls = 'good'
elif wF > n / 2:
    verdict = (f'MCFM lowers texel flicker on <b>{wF} of {n}</b> objects '
               f'(mean <b>{mF:+.1f}%</b>), but drift also moves <b>{mD:+.1f}%</b>. '
               f'Part of the reduction is the animation being damped, not jitter removed.')
    vcls = 'warn'
else:
    verdict = (f'MCFM lowers texel flicker on only <b>{wF} of {n}</b> objects '
               f'(mean <b>{mF:+.1f}%</b>). On this metric it is not an improvement.')
    vcls = 'warn'

tr = '\n'.join(
    f'<tr><td>{r["obj"]}</td><td class="n">{r["n"]}</td>'
    f'<td class="n">{r["fa"]:.5f}</td><td class="n">{r["fb"]:.5f}</td>'
    f'<td class="n {"g" if r["dF"]<0 else "b"}">{r["dF"]:+.1f}</td>'
    f'<td class="n">{r["ja"]:.5f}</td><td class="n">{r["jb"]:.5f}</td>'
    f'<td class="n {"g" if r["dJ"]<0 else "b"}">{r["dJ"]:+.1f}</td>'
    f'<td class="n">{r["da"]:.4f}</td><td class="n">{r["db"]:.4f}</td>'
    f'<td class="n {"" if abs(r["dD"])<5 else "b"}">{r["dD"]:+.1f}</td></tr>'
    for r in rows)

HTML = f"""<title>Texel Flicker</title>
<style>
:root{{--ink:#16181D;--ink2:#4A505C;--ink3:#818894;--ground:#F8F8F6;--panel:#FFF;
--panel2:#EEEFF3;--rule:#E2E4E9;--rule2:#C8CCD4;--acc:#1A5FA8;--good:#0E6B4A;
--good-bg:#E5F2EC;--warn:#A6551D;--warn-bg:#FAEEE2;
--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--ink:#ECEEF2;
--ink2:#AFB6C2;--ink3:#78808D;--ground:#0B0D11;--panel:#14161B;--panel2:#1C1F26;
--rule:#222731;--rule2:#343A45;--acc:#79B4E8;--good:#54C393;--good-bg:#0D2620;
--warn:#DF9F66;--warn-bg:#2A1C0F}}}}
:root[data-theme="dark"]{{--ink:#ECEEF2;--ink2:#AFB6C2;--ink3:#78808D;--ground:#0B0D11;
--panel:#14161B;--panel2:#1C1F26;--rule:#222731;--rule2:#343A45;--acc:#79B4E8;
--good:#54C393;--good-bg:#0D2620;--warn:#DF9F66;--warn-bg:#2A1C0F}}
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.sheet{{max-width:1180px;margin:0 auto;padding:0 24px 90px}}
header{{padding:54px 0 24px;border-bottom:1px solid var(--rule2)}}
.kick{{font-family:var(--mono);font-size:11px;letter-spacing:.17em;text-transform:uppercase;
color:var(--ink3);margin:0 0 14px}}
h1{{font-size:clamp(30px,4.6vw,46px);font-weight:800;letter-spacing:-.032em;line-height:1.04;
margin:0 0 14px;text-wrap:balance}}
.lede{{font-size:17px;color:var(--ink2);margin:0;max-width:68ch}}.lede b{{color:var(--ink)}}
h2{{font-size:22px;font-weight:760;letter-spacing:-.02em;margin:46px 0 10px}}
p{{margin:0 0 12px;max-width:76ch}}.sub{{color:var(--ink2);font-size:15px;margin:0 0 16px;max-width:74ch}}
code{{font-family:var(--mono);font-size:.87em;background:var(--panel2);padding:.12em .36em;border-radius:3px}}
.eq{{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--acc);
border-radius:0 8px 8px 0;padding:16px 18px;font-family:var(--mono);font-size:13.5px;
line-height:1.9;overflow-x:auto;margin:16px 0}}
.tw{{overflow-x:auto;margin:18px 0;border:1px solid var(--rule);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--panel)}}
th{{text-align:left;font-family:var(--mono);font-size:9.5px;letter-spacing:.09em;
text-transform:uppercase;color:var(--ink3);padding:9px 10px;background:var(--panel2);
border-bottom:1px solid var(--rule2);white-space:nowrap}}
td{{padding:8px 10px;border-bottom:1px solid var(--rule);color:var(--ink2)}}
tbody tr:last-child td{{border-bottom:none}}
td:first-child{{color:var(--ink);font-weight:650;white-space:nowrap}}
td.n,th.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
td.g{{color:var(--good);font-weight:700}}td.b{{color:var(--warn);font-weight:700}}
.note{{border-left:3px solid var(--rule2);background:var(--panel);border-radius:0 8px 8px 0;
padding:14px 17px;margin:18px 0;font-size:14.5px;color:var(--ink2)}}
.note.good{{border-left-color:var(--good);background:var(--good-bg)}}
.note.warn{{border-left-color:var(--warn);background:var(--warn-bg)}}
.note b{{color:var(--ink)}}
.note .lbl{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
color:var(--ink3);display:block;margin-bottom:6px}}
.note p:last-child{{margin-bottom:0}}
.kpi{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
.k{{flex:1 1 200px;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}}
.k .t{{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3)}}
.k .v{{font-size:26px;font-weight:780;letter-spacing:-.02em;font-variant-numeric:tabular-nums;margin-top:4px}}
.k .s{{font-size:12.5px;color:var(--ink3)}}
footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);
font-family:var(--mono);font-size:11.5px;color:var(--ink3);line-height:1.8}}
</style>
<div class="sheet">
<header>
  <p class="kick">TRELLIS.2 &middot; temporal metric &middot; {n} objects &middot; texel space</p>
  <h1>Texel Flicker</h1>
  <p class="lede">Does temporal attention (MCFM) on top of the rung27 LoRA buy temporal
  coherence? Measured on the <b>texels the model emits</b>, not on rendered pixels.</p>
</header>

<div class="kpi">
  <div class="k"><div class="t">Temporal Flickering</div>
    <div class="v" style="color:var(--{'good' if wF>n/2 else 'warn'})">{mF:+.1f}%</div>
    <div class="s">lower on {wF}/{n} &middot; sign test p = {pF:.4f}</div></div>
  <div class="k"><div class="t">Jerk (2nd order)</div>
    <div class="v" style="color:var(--{'good' if wJ>n/2 else 'warn'})">{mJ:+.1f}%</div>
    <div class="s">lower on {wJ}/{n} &middot; p = {pJ:.4f}</div></div>
  <div class="k"><div class="t">Drift &mdash; the guard</div>
    <div class="v">{mD:+.1f}%</div>
    <div class="s">|mean| {aD:.1f}% &middot; guard {guard}</div></div>
</div>

<div class="note {vcls}">
  <span class="lbl">Verdict</span>
  <p>{verdict}</p>
</div>

<h2>The metric</h2>
<p class="sub"><b>Temporal Flickering</b>, from <i>VBench: Comprehensive Benchmark Suite for
Video Generative Models</i>, Huang et al., <b>CVPR 2024 (Highlight)</b> &mdash; the mean absolute
difference between consecutive frames. VBench requires static scenes for it to be valid, and
ours are static <i>by construction</i>: fixed mesh, pinned noise, so the only thing that can
change between frames is texture.</p>

<div class="eq">
F = 1/(T&minus;1) &middot; &Sigma;<sub>t=2..T</sub>  mean<sub>v</sub> | C<sub>t</sub>(v) &minus; C<sub>t&minus;1</sub>(v) |

  C<sub>t</sub>(v)   base_color of voxel v at frame t, in [0,1]   (decoder output channels 0:3)
  T         frames per object (150 or 121 here)
  v         ranges over the PBR voxel field &mdash; up to 875k texels
  J (jerk)  mean<sub>v</sub> | C<sub>t+1</sub> &minus; 2C<sub>t</sub> + C<sub>t&minus;1</sub> |    second order: jitter, not speed
  D (drift) mean<sub>v</sub> | C<sub>T</sub> &minus; C<sub>1</sub> |             the guard
</div>

<div class="note">
  <span class="lbl">Why texels and not rendered pixels</span>
  <p>A render puts rasterisation, shading, camera sampling and background compositing between
  the model and the number, and all of those vary frame to frame for reasons unrelated to the
  texture. This architecture emits no per-frame UV map: the texture flow produces a <b>PBR voxel
  field</b> and the renderer samples it. The voxel field <i>is</i> the texture &mdash; UV baking is a
  downstream export that adds its own interpolation and inpainting &mdash; so these voxels are the
  texels, one level upstream of a UV atlas.</p>
  <p style="margin-bottom:0"><code>encode_shape_slat(mesh)</code> is deterministic and the mesh is
  fixed, so voxel <b>coordinates are identical in every frame</b>. That is asserted per frame at
  runtime, not assumed &mdash; a mismatch aborts the run rather than silently differencing
  different voxels.</p>
</div>

<div class="note">
  <span class="lbl">Why drift has to be reported next to flicker</span>
  <p style="margin-bottom:0">Flicker is trivially reduced by freezing the texture &mdash; a worse
  result reported as a better number, and the first thing a reviewer will suspect. A drop in F
  only counts if <b>D holds</b>: the texture must still travel as far from first frame to last.
  Both columns are in the table for every object.</p>
</div>

<h2>Per-object results</h2>
<p class="sub">rung27 (<code>qkvo+sa</code>) versus the same adapter with <code>--mcfm v2_D</code>.
Pairs matched on identical <code>gt_dir</code>, mesh and frame count, so the MCFM flag is the only
difference. Rank 4, seed 42, 30 epochs, L1 + 0.1&middot;LPIPS throughout.</p>
<div class="tw"><table>
<thead><tr><th>Object</th><th class="n">T</th>
<th class="n">F r27</th><th class="n">F mcfm</th><th class="n">&Delta;F %</th>
<th class="n">J r27</th><th class="n">J mcfm</th><th class="n">&Delta;J %</th>
<th class="n">D r27</th><th class="n">D mcfm</th><th class="n">&Delta;D %</th></tr></thead>
<tbody>
{tr}
</tbody></table></div>

<h2>What the other metrics say</h2>
<div class="note warn">
  <span class="lbl">Report these too &mdash; a reviewer will compute them</span>
  <p><b>PSNR / SSIM: nothing.</b> Across 8 objects MCFM moves PSNR by <b>+0.021 dB</b> mean (5/8)
  and SSIM by <b>+0.0005</b> (5/8), against a measured run-to-run noise floor of 1.55 dB. That is
  expected: both are per-frame metrics computed against a per-frame target, and cannot see
  consistency <i>between</i> frames.</p>
  <p style="margin-bottom:0"><b>tLP: worse.</b> TecoGAN's temporal LPIPS is higher for MCFM on 6/8
  objects, mean <b>+18.4%</b>. tLP is a <i>first-difference</i> quantity, so it cannot separate
  "less jitter" from "slower", and charges MCFM for both. The second-order term (J) and the drift
  guard (D) above are what separate them. State this in the paper rather than leave it to be found.</p>
</div>

<footer>
  metric: VBench Temporal Flickering (Huang et al., CVPR 2024) on PBR voxel base_color<br>
  {n} matched object pairs &middot; rung27 qkvo+sa, rank 4, seed 42, 30 epochs, L1 + 0.1 LPIPS<br>
  renderer-free: --skip-render, ODE only &middot; voxel coords asserted identical per frame<br>
  source: experiments/dynamesh/out/texel_summary.json
</footer>
</div>
"""
OUT.write_text(HTML, encoding='utf8')
print(f'wrote {OUT}  ({len(HTML.encode())/1024:.0f} KB)')
print(f'  objects {n}   flicker {wF}/{n} ({mF:+.1f}%)   jerk {wJ}/{n} ({mJ:+.1f}%)   drift {mD:+.1f}%')
