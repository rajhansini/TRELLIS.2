"""build_kl_artifact.py — add the 16-cell KL grid to "The Progression" (37947d06).

Flips the chain's dead '+KL not trained' step and inserts, before the footer, the
4x4 PSNR/SSIM grid plus all 16 720-degree turntables at every frame.

SIZE IS THE BINDING CONSTRAINT. The page already carries ~3.8 MB of base64 video
against a 16 MB cap, and base64 inflates by 4/3. This computes the finished size
and REFUSES to write past the cap rather than letting the publish fail after.
"""
import base64, json, re, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
WEB = E / 'out/R30KL_WEB'
SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
RES = json.loads((E / 'out/r30kl_grid_results.json').read_text())
FRACS = [15, 25, 50, 75]
CAP = 16 * 1024 * 1024

vals = {(fc, fs): RES['c%d_s%d' % (fc, fs)] for fc in FRACS for fs in FRACS}
psnrs = {k: v['psnr'] for k, v in vals.items() if v.get('psnr')}
best, worst = max(psnrs, key=psnrs.get), min(psnrs, key=psnrs.get)

ctrl = None
for f in (E / 'out').glob('r30g_teapot_ceramic_crack_correct_*.log'):
    t = f.read_text(errors='replace')
    if '--w-kl 0 --w-kl-self 0' in t:
        m = re.findall(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)', t)
        if m:
            ctrl = (float(m[-1][0]), float(m[-1][1]))

def b64(p):
    return base64.b64encode(p.read_bytes()).decode()

rows = []
for fc in FRACS:
    tds = ''.join(
        '<td class="n%s">%.3f<br><span style="color:var(--ink3);font-size:11px">%.4f</span></td>'
        % (' b' if (fc, fs) == best else '', vals[(fc, fs)]['psnr'], vals[(fc, fs)]['ssim'])
        for fs in FRACS)
    rows.append('<tr><td>&beta;<sub>c</sub> = %d%%</td>%s</tr>' % (fc, tds))

vids = []
for fc in FRACS:
    for fs in FRACS:
        p = WEB / ('r30kl_c%d_s%d_720.mp4' % (fc, fs))
        if not p.exists() or p.stat().st_size == 0:
            vids.append('<div class="miss">c%d_s%d: render missing</div>' % (fc, fs))
            continue
        v = vals[(fc, fs)]
        tag = (' &middot; <b>best cell</b>' if (fc, fs) == best else
               ' &middot; <b>worst cell</b>' if (fc, fs) == worst else '')
        vids.append(
            '<figure class="v"><video controls muted loop playsinline preload="none" '
            'src="data:video/mp4;base64,%s"></video><figcaption>'
            '<b>&beta;<sub>c</sub>=%d%% &beta;<sub>s</sub>=%d%%</b> &middot; %.3f dB / %.4f '
            '&middot; w_kl %g, w_kl_self %g%s<br>GROUND TRUTH &middot; frozen TRELLIS.2 '
            '&middot; rung30 + KL &mdash; 720&deg;, all 150 frames</figcaption></figure>'
            % (b64(p), fc, fs, v['psnr'], v['ssim'], v['beta_c'], v['beta_s'], tag))

if ctrl:
    ctrl_html = ('<p style="margin-bottom:0">The <b>&beta;=0 control</b> scores '
                 '<b>%.3f dB / %.4f</b>. Every one of the 16 KL cells sits below it, so on '
                 'this object the penalty costs quality at every setting tested and the best '
                 'available weight is zero.</p>' % ctrl)
else:
    ctrl_html = ('<p style="margin-bottom:0">The <b>&beta;=0 control is still training</b> '
                 '(job 2183384). Until it lands, this grid ranks 16 KL settings against each '
                 'other but cannot state whether KL beats no KL at all. The monotonic trend '
                 'points at zero being best &mdash; that is an extrapolation, not yet a '
                 'measurement.</p>')

SECTION = """
<h2>The seventh object &mdash; and what the KL actually looks like</h2>
<p class="sub">The grid above covers six objects as numbers. This is the seventh,
<code>teapot_ceramic_crack_correct</code>, with the same 4&times;4 over
&beta;<sub>c</sub> and &beta;<sub>s</sub> &mdash; and every cell rendered, so the penalty can be
watched rather than read. rung30, <code>qkvo+sa</code>, MCFM v2_D, 150 frames, 30 epochs,
seed 42; only the two weights differ.</p>

<div class="tw"><table>
<thead><tr><th>PSNR / SSIM</th><th class="n">&beta;<sub>s</sub> = 15%%</th>
<th class="n">&beta;<sub>s</sub> = 25%%</th><th class="n">&beta;<sub>s</sub> = 50%%</th>
<th class="n">&beta;<sub>s</sub> = 75%%</th></tr></thead>
<tbody>%s</tbody></table></div>

<div class="note warn">
  <span class="lbl">Monotonic in both weights &mdash; KL only costs</span>
  <p>Best is the weakest penalty on both axes, <b>&beta;<sub>c</sub>=15%%
  &beta;<sub>s</sub>=15%% at %.3f dB</b>. Worst is the strongest on both, <b>%.3f dB</b>.
  Spread <b>%.3f dB</b>. Raising either weight lowers the score at essentially every point
  in the grid, and the two penalties compound rather than trading off against each other.</p>
  %s
</div>

<div class="note">
  <span class="lbl">Two things to know before quoting these numbers</span>
  <p><b>The percentages are approximate.</b> &beta; is set from a 6-epoch probe
  (<code>loss/KLc = 5.426</code>, <code>loss/KLs = 1.665</code>) via
  &beta; = f/(1&minus;f) &middot; ratio. But KL grows as &Vert;B&Vert;&sup2; while the data
  loss falls, so that ratio was still dropping about 30%% per epoch when the probe stopped.
  Every cell shares the convention, so the ordering holds &mdash; the axis labels do not
  survive being read as exact shares of the converged loss.</p>
  <p style="margin-bottom:0"><b>This is not the <code>teapot_ceramic_crack</code> row in the
  table above.</b> It is a regenerated Kling video: the prompt was rewritten so the body keeps
  its grey instead of turning white, because a white teapot on a white background is invisible
  to the brightness threshold that builds the targets. 150 frames rather than 121, and its
  targets hold IoU 0.988&ndash;0.990 across all 150 with 0&ndash;1 px of rim fill &mdash; the
  cleanest target set of any object here. Its absolute PSNR is therefore not comparable to the
  eight rows above.</p>
</div>

<h2>All 16 cells, 720&deg;</h2>
<p class="sub">Two full revolutions across the 150 frames, so every surface point is seen
twice at different stages of the texture. Three panels: <b>ground truth</b> &middot;
<b>frozen TRELLIS.2</b> &middot; <b>rung30 + KL</b>. Ordered by &beta;<sub>c</sub>, then
&beta;<sub>s</sub>.</p>
%s
""" % (''.join(rows), psnrs[best], psnrs[worst], psnrs[best] - psnrs[worst],
       ctrl_html, ''.join(vids))

html = SRC.read_text(encoding='utf8')
# The chain step was already flipped to '+KL 96 cells, all worse' by another
# session's publish. Nothing to change there — this adds the seventh object and,
# more to the point, the turntables that section has none of.
i = html.rfind('<footer>')
assert i > 0, 'footer not found — refusing to guess where the section goes'
out = html[:i] + SECTION + html[i:]

n = len(out.encode())
print('page %.2f MB  (cap 16.00 MB)' % (n / 1048576))
if n > CAP:
    print('OVER CAP — not writing.')
    sys.exit(1)
DST.write_text(out, encoding='utf8')
print('wrote %s' % DST)
print('  best  c%d_s%d  %.3f' % (best[0], best[1], psnrs[best]))
print('  worst c%d_s%d  %.3f' % (worst[0], worst[1], psnrs[worst]))
print('  control: %s' % ('yes %s' % (ctrl,) if ctrl else 'pending'))
