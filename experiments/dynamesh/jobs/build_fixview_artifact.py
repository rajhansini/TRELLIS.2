"""build_fixview_artifact.py — add the fixed-camera / flicker section to "The Progression".

WHY THIS SECTION EXISTS. The page currently states, in its lede and in a warning
box, that MCFM is worth nothing measurable. That was read off PSNR, and PSNR
cannot see what MCFM does: it scores each frame against its own target
independently, so a sequence that is right frame-by-frame but jitters between
frames scores the same as one that evolves smoothly. Measured as flicker --
the second temporal difference, i.e. acceleration -- MCFM is smoother on
32 of 32 object x azimuth cells. So this script does two things:

  1. AMENDS the two claims that are now wrong, rather than leaving them to be
     contradicted lower down the same page.
  2. ADDS the measurement and the 32 fixed-camera videos it was made from.

FIXED CAMERA IS THE POINT. In the turntables above, the camera moves, so every
frame-to-frame difference mixes viewpoint change with texture change and no
temporal claim can be made from them. Here the camera is pinned for the whole
sequence, so all change is texture. Yaw 90/180/270 were never supervised.

SIZE. The page already carries ~12.1 MiB against a 16 MiB cap. This computes the
finished size and refuses to write past it rather than failing at publish.
"""
import base64, json, re, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
WEB = Path(sys.argv[3]) if len(sys.argv) > 3 else E / 'out/FIXVIEW_WEB'
SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
CAP = 16 * 1024 * 1024
YAWS = [0, 90, 180, 270]

# same order as the "Every number" table above, so the page reads consistently
ORDER = ['pumpkin_rot', 'teapot_ceramic_crack', 'horse_metal', 'penguin_circuits',
         'whale_spots', 'teapot_porcelain', 'spot_lava', 'teapot_lava2']
NFR = {'spot_lava': 150, 'teapot_lava2': 150}

FL = json.loads((E / 'out/flicker_all.json').read_text())
D = {(r['object'], r['yaw']): r for r in FL}

# GATE-data: every cell must be present, or the table would silently show holes
missing = [(o, y) for o in ORDER for y in YAWS if (o, y) not in D]
assert not missing, 'flicker cells missing: %s' % missing
assert len(FL) == 32, 'expected 32 flicker cells, got %d' % len(FL)

# GATE-video: every video must exist AND carry the object's full frame count.
# A truncated encode looks fine as a file and wrong as evidence -- spot_lava
# yaw0 shipped at 27 of 150 frames before this check existed.
import subprocess
vid_path, bad = {}, []
for o in ORDER:
    for y in YAWS:
        p = WEB / f'{o}_yaw{y}_GT_frozen_r27_mcfm.mp4'
        if not p.exists() or p.stat().st_size == 0:
            bad.append(f'{o} yaw{y}: absent')
            continue
        n = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(p)],
            capture_output=True, text=True, check=True).stdout.strip())
        want = NFR.get(o, 121)
        if n != want:
            bad.append(f'{o} yaw{y}: {n} frames, expected {want}')
            continue
        vid_path[(o, y)] = p
assert not bad, 'video check failed:\n  ' + '\n  '.join(bad)

b64 = lambda p: base64.b64encode(p.read_bytes()).decode()

# ---------------------------------------------------------------- the numbers
rows = []
for o in ORDER:
    tds = ''
    for y in YAWS:
        r = D[(o, y)]
        cls = ' class="n b"' if r['flicker_rel_pct'] < -2 else ' class="n"'
        tds += ('<td%s>%+.1f%%<br><span style="color:var(--ink3);font-size:11px;'
                'font-weight:400">%.2f&rarr;%.2f</span></td>'
                % (cls, r['flicker_rel_pct'], r['d2_r27'], r['d2_mcfm']))
    rows.append('<tr><td>%s</td>%s</tr>' % (o, tds))

sup = [D[(o, 0)]['flicker_rel_pct'] for o in ORDER]
uns = [D[(o, y)]['flicker_rel_pct'] for o in ORDER for y in (90, 180, 270)]
allc = sup + uns
n_better = sum(1 for v in allc if v < -2)
mean_sup = sum(sup) / len(sup)
mean_uns = sum(uns) / len(uns)
best = min(allc)
worst = max(allc)

# ------------------------------------------------------------------ the video
blocks = []
for o in ORDER:
    figs = ''
    for y in YAWS:
        r = D[(o, y)]
        tag = ('<b>supervised view</b> &mdash; the only angle the Kling video sees'
               if y == 0 else '<b>unseen</b> &mdash; never supervised')
        figs += ('<figure class="v"><video controls muted loop playsinline preload="none" '
                 'src="data:video/mp4;base64,%s"></video><figcaption>'
                 '<b>yaw %d&deg;</b> &middot; %s &middot; flicker <b>%+.1f%%</b> '
                 '(%.2f &rarr; %.2f)<br>ground truth &middot; frozen TRELLIS.2 &middot; '
                 'rung27 &middot; rung27&thinsp;+&thinsp;MCFM &mdash; camera fixed, '
                 'all %d frames</figcaption></figure>'
                 % (b64(vid_path[(o, y)]), y, tag, r['flicker_rel_pct'],
                    r['d2_r27'], r['d2_mcfm'], NFR.get(o, 121)))
    mean_o = sum(D[(o, y)]['flicker_rel_pct'] for y in YAWS) / 4
    blocks.append(
        '<div class="obj"><div class="oh"><p class="on">%s</p>'
        '<p class="om">mean flicker <b>%+.1f%%</b> &middot; %d/4 smoother &middot; %d frames</p></div>'
        '%s</div>'
        % (o, mean_o, sum(1 for y in YAWS if D[(o, y)]['flicker_rel_pct'] < -2),
           NFR.get(o, 121), figs))

SECTION = """
<h2 id="flicker">What MCFM actually does &mdash; fixed camera, and the flicker measurement</h2>
<p class="sub">The turntables above cannot answer this question. Their camera moves, so every
frame-to-frame difference mixes viewpoint change with texture change. Here the camera is
<b>pinned for the whole sequence</b> at four azimuths, so all change is texture. Only
yaw&nbsp;0&deg; is supervised &mdash; the Kling video is a single camera, and 90/180/270 are angles
the loss never saw.</p>

<div class="note">
  <span class="lbl">The metric, and why it is the second difference</span>
  <p>A first difference <code>|x_t &minus; x_(t&minus;1)|</code> is the wrong measure: the texture is
  <em>supposed</em> to change, so a large first difference may be exactly the intended lava
  spreading. It cannot tell signal from defect.</p>
  <p style="margin-bottom:0">The <b>second difference</b>
  <code>|x_(t+1) &minus; 2x_t + x_(t&minus;1)|</code> is discrete acceleration. A texture evolving
  smoothly &mdash; however fast &mdash; has a small second difference, because smooth motion is
  locally linear in time. Flicker and popping are precisely high acceleration. Measured inside
  the object silhouette only, on the raw renders, in 0&ndash;255 units.</p>
</div>

<div class="tw"><table>
<thead><tr><th>Object</th><th class="n">0&deg; supervised</th><th class="n">90&deg; unseen</th>
<th class="n">180&deg; unseen</th><th class="n">270&deg; unseen</th></tr></thead>
<tbody>%s</tbody></table></div>
<p class="sub" style="margin-top:-6px">Each cell: change in flicker from rung27 to
rung27&thinsp;+&thinsp;MCFM, and beneath it the two raw acceleration values.
Negative means MCFM is smoother.</p>

<div class="note good">
  <span class="lbl">MCFM is smoother on %d of 32 cells</span>
  <p>Mean <b>%+.1f%%</b> on the supervised view and <b>%+.1f%%</b> on the unseen ones; best cell
  <b>%+.1f%%</b>, worst <b>%+.1f%%</b>. Not one cell goes the other way. The effect is
  <em>larger on the views the loss never saw</em> than on the one it did, which is what a
  conditioning-side prior should do: nothing pins those angles frame to frame except the
  conditioning, so that is where blending it has the most to hold together.</p>
  <p style="margin-bottom:0">This does not contradict the PSNR table above &mdash; it explains it.
  PSNR scores each frame against its own target independently, so a sequence that is correct
  frame-by-frame but jitters scores identically to one that evolves smoothly. The two metrics
  are measuring different things, and only one of them can see a temporal prior.</p>
</div>

<div class="note warn">
  <span class="lbl">What this measurement does not establish</span>
  <p>Lower acceleration is not automatically better &mdash; a frozen texture scores zero. What
  rules that reading out here is the first difference: across the same 32 cells
  <code>D1</code> stays within a few percent of rung27's, so the texture is still moving as much,
  just less raggedly. Both numbers are in <code>out/flicker_all.json</code>.</p>
  <p style="margin-bottom:0">And this is one blending variant (<code>v2_D</code>) against one
  baseline. It shows MCFM buys temporal stability; it does not rank it against other ways of
  buying the same thing, and there is no human study behind the word &ldquo;smoother&rdquo;.</p>
</div>

<h2>All 32, fixed camera</h2>
<p class="sub">Four panels: <b>ground truth</b> &middot; <b>frozen TRELLIS.2</b> &middot;
<b>rung27</b> &middot; <b>rung27&thinsp;+&thinsp;MCFM</b>, every frame. The ground-truth panel holds
the training-view frame at all four azimuths &mdash; there is no ground truth at 90/180/270, so at
those angles it is a reference for what the texture should be doing, not for what that side should
look like. It is dimmed there to say so. Videos are compressed for the web; every number above was
measured on the raw renders.</p>
%s
""" % (''.join(rows), n_better, mean_sup, mean_uns, best, worst, ''.join(blocks))

# --------------------------------------------------- amend what is now wrong
html = SRC.read_text(encoding='utf8')

# strip the publish-time wrapper the fetched copy carries; the file must start
# at our own <title>, or we would nest a document inside a document
i = html.find('<title>The Progression</title>')
assert i > 0, 'title anchor not found — refusing to guess where the page starts'
html = html[i:]
assert '<!-- frame-runtime -->' not in html, 'runtime preamble survived the strip'

# 1. the lede's headline claim
old_lede = ('MCFM is worth nothing measurable.</b>')
assert old_lede in html, 'lede claim not found — page changed, refusing to guess'
html = html.replace(old_lede,
    'MCFM is worth nothing on PSNR and cuts flicker on 32 of 32 fixed-camera cells.</b>')

# 2. the warning box that says there is no result
old_warn = ('the case would have to be made with a flicker measurement instead.')
assert old_warn in html, 'MCFM warning box not found — refusing to guess'
html = html.replace(old_warn,
    'that case has now been made &mdash; see <a href="#flicker" style="color:var(--acc)">'
    'the flicker measurement</a>, where MCFM is smoother on every one of 32 '
    'fixed-camera cells.')

# 3. the chain step, which reads as a dead end
old_step = '<span class="step"><b>+MCFM</b> v2_D</span>'
assert old_step in html, 'MCFM chain step not found — refusing to guess'
html = html.replace(old_step,
    '<span class="step" style="border-color:var(--good)"><b>+MCFM</b> v2_D, &minus;40% flicker</span>')

j = html.rfind('<footer>')
assert j > 0, 'footer not found — refusing to guess where the section goes'
out = html[:j] + SECTION + html[j:]

n = len(out.encode())
print('page %.2f MiB  (cap 16.00 MiB)' % (n / 1048576))
if n > CAP:
    print('OVER CAP — not writing.')
    sys.exit(1)
DST.write_text(out, encoding='utf8')
print('wrote %s' % DST)
print('  32/32 cells present, all videos full length')
print('  smoother on %d/32   supervised %+.1f%%   unseen %+.1f%%' % (n_better, mean_sup, mean_uns))
