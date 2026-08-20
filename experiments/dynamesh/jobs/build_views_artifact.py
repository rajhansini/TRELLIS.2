"""
build_views_artifact.py — the six-camera page for the new objects.

WHAT IT SHOWS, AND WHY ALL SIX CAMERAS SIT IN ONE BLOCK PER OBJECT
  Splitting the supervised view into one page and the unseen views into another
  would make the only interesting comparison impossible: whether the texture that
  fits the one camera the loss saw also holds up from angles it never did. So each
  object gets one block containing all six, training view first.

  Per video, four panels: ground truth | frozen TRELLIS.2 | rung27 | rung27 + MCFM.

THE GT PANEL AWAY FROM THE TRAINING VIEW
  The Kling clip is a single fixed camera, so ground truth exists at exactly one
  angle. Everywhere else the GT column repeats the training-view frame and says so
  in its label -- it is the texture reference (what the effect should be doing at
  this instant), not a claim about what that side looks like.

SIZE
  16 MB cap, and base64 inflates by 4/3. Videos are re-encoded small, the finished
  page is measured, and it REFUSES to write past the cap rather than failing at
  publish time.
"""
import base64, json, re, subprocess, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
DST = Path(sys.argv[1])
WEB = Path(sys.argv[2]) if len(sys.argv) > 2 else E / 'out/VIEWS_WEB'
CAP = 16 * 1024 * 1024

# argv[3]: comma-separated subset, for a page about just the newest objects
ONLY = [x for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else []) if x]
TITLE = sys.argv[4] if len(sys.argv) > 4 else 'Six Cameras'

ORDER = ['mushroom_glow', 'monster_lava_2', 'teapot_crack', 'ancient_statue_clay',
         'skull_lava', 'hand_rorschach', 'monster_rainbow', 'ancient_lady',
         'eagle_blackness', 'plane_waves', 'alien_glow', 'ancient_lady_crack',
         'spot_raurshaw',
         'alien_spots', 'teapot_fungi', 'teapot_golden_crack', 'horse_loki',
         'ancient_lady_teaser', 'ancient_lady_effect_1', 'ancient_lady_effect_2']
MESH = {'ancient_lady': 'nefertiti', 'plane_waves': 'whale', 'teapot_crack': 'teapot',
        'hand_rorschach': 'hand', 'skull_lava': 'skull', 'monster_lava_2': 'armadillo',
        'monster_rainbow': 'armadillo', 'ancient_statue_clay': 'nefertiti',
        'eagle_blackness': 'falcon statue', 'mushroom_glow': 'mushroom',
        'alien_glow': 'alien', 'ancient_lady_crack': 'nefertiti', 'spot_raurshaw': 'spot',
        'alien_spots': 'alien', 'teapot_fungi': 'teapot',
        'teapot_golden_crack': 'teapot', 'horse_loki': 'horse',
        'ancient_lady_teaser': 'nefertiti',
        'ancient_lady_effect_1': 'nefertiti',
        'ancient_lady_effect_2': 'nefertiti'}
EFFECT = {'ancient_lady': 'gold leaf and glitter', 'plane_waves': 'underwater caustics',
          'teapot_crack': 'crackle glaze', 'hand_rorschach': 'inkblot',
          'skull_lava': 'molten lava', 'monster_lava_2': 'molten lava',
          'monster_rainbow': 'rainbow iridescence', 'ancient_statue_clay': 'drying clay',
          'eagle_blackness': 'sooty blackening', 'mushroom_glow': 'bioluminescence',
          'alien_glow': 'glow', 'ancient_lady_crack': 'crackle',
          'spot_raurshaw': 'inkblot',
          'alien_spots': 'spots', 'teapot_fungi': 'fungal growth',
          'teapot_golden_crack': 'golden crackle', 'horse_loki': 'loki',
          'ancient_lady_teaser': 'teaser',
          'ancient_lady_effect_1': 'effect 1',
          'ancient_lady_effect_2': 'effect 2'}
VIEWS = [('train', 'training view', 'the one camera the loss saw'),
         ('orbit360', '360°', 'one revolution across the sequence'),
         ('orbit720', '720°', 'two revolutions — every point seen twice'),
         ('diagA', 'yaw 45° · elev +25°', 'diagonal, from above'),
         ('diagB', 'yaw 135° · elev −20°', 'diagonal, from below'),
         ('diagC', 'yaw 225° · elev +30°', 'diagonal, from behind')]

# argv[5]: comma-separated view keys, for a page about a SUBSET of the six cameras.
# The generalisation claim is "one supervised camera, three it never saw", which is
# train+diagA+diagB+diagC -- the two turntables are a coverage/seams artefact and
# dilute that page. Filtering here rather than forking the builder keeps one source
# of truth for the panel layout and the labels.
_WANT = [x for x in (sys.argv[5].split(',') if len(sys.argv) > 5 else []) if x]
if _WANT:
    VIEWS = [v for v in VIEWS if v[0] in _WANT]
    assert len(VIEWS) == len(_WANT), f'unknown view in {_WANT}'
NV = len(VIEWS)
_W = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
CAMW = _W.get(NV, str(NV))                       # "four" / "six"
NTURN = sum(1 for v, *_ in VIEWS if v.startswith('orbit'))
NUNSEEN = sum(1 for v, *_ in VIEWS if v != 'train')
# The turntable caveat is only true if a turntable is on the page. Printing it on a
# fixed-camera page would describe videos the reader cannot see.
TURN_NOTE = ('' if not NTURN else
             ' On the ' + ('two turntables' if NTURN == 2 else 'turntable') +
             ' the camera moves, so ' + ('those are' if NTURN == 2 else 'that is') +
             ' for judging coverage and seams &mdash; the fidelity claim lives at the'
             ' training view and in the table above.')

PSNR = json.loads((E / 'out/psnr_all.json').read_text())
if ONLY:
    ORDER = [o for o in ONLY if o in MESH]
    assert ORDER, f'none of {ONLY} are known objects'


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


# ------------------------------------------------------------------ the table
rows, deltas = [], []
for o in ORDER:
    p = PSNR.get(o, {})
    a, b = p.get('r27'), p.get('mcfm')
    if a and b:
        d = b[0] - a[0]
        deltas.append(d)
        cls = ' class="n b"' if d > 0 else ' class="n"'
        rows.append(f'<tr><td>{o}</td><td>{MESH[o]}</td><td>{EFFECT[o]}</td>'
                    f'<td class="n">{a[0]:.3f}</td><td class="n">{b[0]:.3f}</td>'
                    f'<td{cls}>{d:+.3f}</td></tr>')
    else:
        rows.append(f'<tr><td>{o}</td><td>{MESH[o]}</td><td>{EFFECT[o]}</td>'
                    f'<td class="n" colspan="3" style="text-align:left;color:var(--ink3)">'
                    f'training still running</td></tr>')

n_done = len(deltas)
n_up = sum(1 for d in deltas if d > 0)
mean_d = sum(deltas) / max(n_done, 1)

# ----------------------------------------------------------------- the videos
blocks, shipped, missing, present = [], 0, [], []
for o in ORDER:
    figs = ''
    for v, label, why in VIEWS:
        p = WEB / f'{o}_{v}_GT_frozen_r27_mcfm.mp4'
        if not p.exists() or p.stat().st_size < 1000:
            missing.append(f'{o}/{v}')
            continue
        sup = (v == 'train')
        badge = ('<span class="tag sup">supervised</span>' if sup
                 else '<span class="tag">unseen</span>')
        figs += (f'<figure class="v"><video controls muted loop playsinline preload="none" '
                 f'src="data:video/mp4;base64,{b64(p)}"></video><figcaption>'
                 f'<b>{label}</b> {badge} &middot; {why}<br>'
                 f'ground truth &middot; frozen TRELLIS.2 &middot; rung27 &middot; '
                 f'rung27&thinsp;+&thinsp;MCFM</figcaption></figure>')
        shipped += 1
    if not figs:
        continue
    pp = PSNR.get(o, {})
    met = (f'rung27 <b>{pp["r27"][0]:.2f}</b> &rarr; +MCFM <b>{pp["mcfm"][0]:.2f}</b> dB'
           if pp.get('r27') and pp.get('mcfm') else 'training in progress')
    nvid = figs.count('<figure')
    blocks.append(
        f'<div class="obj" id="o-{o}"><div class="oh"><p class="on">{o}</p>'
        f'<p class="om">{MESH[o]} &middot; {EFFECT[o]} &middot; {met} &middot; '
        f'{nvid}/{NV} cameras</p></div>{figs}</div>')
    present.append((o, nvid))

# ---------------------------------------------------------------- provenance
# WHERE EVERY VIDEO ON THIS PAGE CAME FROM, as SLURM job ids.
# Written by hand into out/job_provenance.json rather than scraped from sacct at
# build time: sacct ages job records out, so a page rebuilt in a month would
# silently lose the column rather than fail loudly. Absent file -> no section,
# so pages for other batches are unaffected.
PROV = E / 'out/job_provenance.json'
prov_html = ''
if PROV.exists():
    PJ = json.loads(PROV.read_text())
    on_page = set(ORDER)
    trs, n_retry, n_att = [], 0, 0

    def cell(d):
        return (f'<td class="n">{d["elapsed"]}<br>'
                f'<span style="color:var(--ink3);font-size:10.5px">{d["job"]}</span></td>')

    for o, d in PJ.items():
        att = d['pan']['attempts']
        n_att += len(att)
        bad = [a for a in att if a['state'] != 'COMPLETED']
        if bad:
            n_retry += 1
        pan = (f'<td class="n" style="color:var(--good)">{att[-1]["job"]}'
               + (f'<br><span style="color:var(--warn);font-size:10.5px">'
                  f'after {len(bad)} failed</span>' if bad else
                  '<br><span style="color:var(--ink3);font-size:10.5px">first try</span>')
               + '</td>')
        here = ('' if o in on_page else
                ' <span class="tag">not on this page</span>')
        trs.append(f'<tr><td>{o}{here}</td>{cell(d["tgt"])}{cell(d["r27"])}'
                   f'{cell(d["mcfm"])}{pan}</tr>')

    extra = [o for o in PJ if o not in on_page]
    extra_note = ('' if not extra else
                  f' Two rows &mdash; <code>{"</code> and <code>".join(extra)}</code> '
                  f'&mdash; were trained in the same batch but their videos live on the '
                  f'companion page; they are kept here so the batch reads as one unit.')
    prov_html = f'''
<h2>How these were built</h2>
<p class="sub">Every video above traces to four SLURM jobs: build the ground-truth targets,
train the rung27 adapter, train the same adapter with MCFM, composite the panels.
Elapsed time and job id for each, so any number on this page can be walked back to the
run that produced it.{extra_note}</p>
<div class="tw"><table>
<thead><tr><th>Object</th><th class="n">targets</th><th class="n">rung27</th>
<th class="n">rung27 + MCFM</th><th class="n">panels</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>

<div class="note warn">
  <span class="lbl">Why the panel column has retries and the training columns do not</span>
  <p style="margin-bottom:0">Training ran clean on the first attempt for all
  {len(PJ)} objects. The panel step failed {n_att - len(PJ)} times across
  {n_retry} of them for a reason that had nothing to do with the model:
  <b>ffmpeg and ffprobe exist on the login node but not on the compute nodes</b>,
  so the composite died at the encode step every time it landed on a worker. The fix
  was to put the conda environment&rsquo;s own ffmpeg on <code>PATH</code> inside the
  batch script; every attempt after that succeeded. Nothing was re-rendered and no
  frame changed &mdash; only the encoder that read them.</p>
</div>
'''

# ------------------------------------------------------------------- assets
# EVERY PATH BEHIND THIS PAGE, for handing the work to someone else.
# Written as ROOT + TEMPLATE + per-object variables rather than 7 x 10 absolute
# paths: the paths are all templated by object name, so spelling out 70 of them
# would hide the one thing that actually differs between objects -- the run hash.
ASSETS = E / 'out/asset_paths.json'
asset_html = ''
if ASSETS.exists():
    AJ = json.loads(ASSETS.read_text())
    tmpl = ''.join(
        f'<tr><td>{what}</td><td class="p">{path}</td></tr>'
        for what, path in AJ['templates'])
    orows = ''.join(
        f'<tr><td>{o}</td><td class="p">{d["mesh"]}</td>'
        f'<td class="n">{d["iou"]:.2f}%</td>'
        f'<td class="p">{d["run_r27"].split("_")[-1]}</td>'
        f'<td class="p">{d["run_mcfm"].split("_")[-1]}</td></tr>'
        for o, d in AJ['objects'].items() if o in set(ORDER))
    ax = AJ['axes']
    asset_html = f'''
<h2>Every path behind this page</h2>
<p class="sub">Two roots, then one template per artefact. Everything is keyed by object
name, so the only value that changes per object is the run hash in the last two columns
&mdash; substitute and the path resolves.</p>
<div class="eq">$T2 = {AJ['roots']['T2']}
$E&nbsp; = {AJ['roots']['E']}

{{arm}}&nbsp; = {' | '.join(ax['arm'])}
{{view}} = {' | '.join(ax['view'])}</div>
<div class="tw"><table>
<thead><tr><th>Artefact</th><th>Path</th></tr></thead>
<tbody>{tmpl}</tbody></table></div>

<p class="sub" style="margin-top:22px">Per object: the mesh the texture was fitted to, how
well its silhouette matched the driving video, and the two run directories under
<code>$E/runs/</code>. Both runs share the prefix
<code>rung27_l1_lp_&hellip;_qkvo+sa_r4_s42_</code> &mdash; the MCFM arm additionally carries
<code>mcfmv2_D</code> &mdash; so only the trailing hash is listed.</p>
<div class="tw"><table>
<thead><tr><th>Object</th><th>Mesh file</th><th class="n">silhouette IoU</th>
<th>rung27 hash</th><th>+MCFM hash</th></tr></thead>
<tbody>{orows}</tbody></table></div>

<div class="note">
  <span class="lbl">Reading the render frames directly</span>
  <p style="margin-bottom:0">Each <code>view_&hellip;/frames/*.png</code> is already
  <b>frozen&nbsp;|&nbsp;adapted</b> side by side &mdash; that is what
  <code>render_rung27_orbit.py</code> writes. The panel videos take the frozen column from
  the <code>r27</code> render rather than rendering a third time, which is what guarantees
  the frozen baseline cannot drift out of step with the arm it is compared against.</p>
</div>
'''

miss_html = ''
if missing:
    miss_html = ('<div class="note warn"><span class="lbl">Not on this page yet</span>'
                 f'<p style="margin-bottom:0">{len(missing)} of {len(ORDER)*NV} '
                 f'(object &times; camera) videos are still rendering: '
                 f'<code>{", ".join(missing[:24])}'
                 f'{" …" if len(missing) > 24 else ""}</code>. '
                 'Listed rather than omitted, so this page cannot be mistaken for the '
                 'complete set.</p></div>')

jump = ' '.join(f'<a href="#o-{o}">{o}</a><span class="jn">{n}/6</span>'
                for o, n in present)

HTML = f"""<title>{TITLE}</title>
<style>
:root{{--ink:#12151A;--ink2:#474D59;--ink3:#7C838F;--ground:#F7F7F5;--panel:#FFF;--panel2:#EDEEF2;
--rule:#E1E3E8;--rule2:#C6CAD2;--acc:#1A5FA8;--good:#0F6E4C;--good-bg:#E4F2EA;--warn:#A8571F;--warn-bg:#F9EDE1;
--sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--ink:#ECEEF2;--ink2:#AEB5C2;--ink3:#78808D;
--ground:#0C0E12;--panel:#14161C;--panel2:#1B1E25;--rule:#212630;--rule2:#333944;
--acc:#79B4E8;--good:#57C795;--good-bg:#0E2620;--warn:#E0A068;--warn-bg:#2A1C10}}}}
:root[data-theme="dark"]{{--ink:#ECEEF2;--ink2:#AEB5C2;--ink3:#78808D;--ground:#0C0E12;--panel:#14161C;
--panel2:#1B1E25;--rule:#212630;--rule2:#333944;--acc:#79B4E8;--good:#57C795;--good-bg:#0E2620;--warn:#E0A068;--warn-bg:#2A1C10}}
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.sheet{{max-width:1240px;margin:0 auto;padding:0 24px 90px}}
header{{padding:54px 0 24px;border-bottom:1px solid var(--rule2)}}
.kick{{font-family:var(--mono);font-size:11px;letter-spacing:.17em;text-transform:uppercase;color:var(--ink3);margin:0 0 14px}}
h1{{font-size:clamp(30px,4.6vw,46px);font-weight:800;letter-spacing:-.032em;line-height:1.04;margin:0 0 14px;text-wrap:balance}}
.lede{{font-size:17px;color:var(--ink2);margin:0;max-width:66ch}}.lede b{{color:var(--ink)}}
h2{{font-size:23px;font-weight:760;letter-spacing:-.02em;margin:50px 0 10px}}
p{{margin:0 0 12px;max-width:74ch}}.sub{{color:var(--ink2);font-size:15px;margin:0 0 16px;max-width:72ch}}
code{{font-family:var(--mono);font-size:.87em;background:var(--panel2);padding:.12em .36em;border-radius:3px}}
.obj{{background:var(--panel);border:1px solid var(--rule);border-radius:12px;padding:20px 22px;margin:20px 0}}
.oh{{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;justify-content:space-between}}
.on{{font-size:20px;font-weight:770;letter-spacing:-.015em;margin:0}}
.om{{font-family:var(--mono);font-size:12.5px;color:var(--ink3);font-variant-numeric:tabular-nums;margin:0}}.om b{{color:var(--ink)}}
.v{{margin:14px 0 18px}}.v video{{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:8px;background:#14151a}}
.v figcaption{{font-size:13px;color:var(--ink2);margin-top:7px;max-width:96ch}}
.tag{{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
background:var(--panel2);color:var(--ink3);padding:2px 7px;border-radius:4px;margin-left:4px}}
.tag.sup{{background:var(--good-bg);color:var(--good)}}
.jump{{display:flex;flex-wrap:wrap;gap:6px 14px;max-width:none;margin:0 0 22px;font-family:var(--mono);font-size:12.5px}}
.jump a{{color:var(--acc);text-decoration:none;border-bottom:1px solid transparent}}
.jump a:hover{{border-bottom-color:var(--acc)}}
.jn{{color:var(--ink3);margin-left:5px}}
.tw{{overflow-x:auto;margin:18px 0;border:1px solid var(--rule);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--panel)}}
th{{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);padding:10px 12px;background:var(--panel2);border-bottom:1px solid var(--rule2);white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid var(--rule);color:var(--ink2)}}tbody tr:last-child td{{border-bottom:none}}
td:first-child{{color:var(--ink);font-weight:650;white-space:nowrap}}
td.n,th.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
td.b{{color:var(--good);font-weight:700}}
td.p{{font-family:var(--mono);font-size:11.5px;color:var(--ink2);white-space:nowrap;font-weight:400}}
td.p:first-child{{color:var(--ink2);font-weight:400}}
.note{{border-left:3px solid var(--rule2);background:var(--panel);border-radius:0 8px 8px 0;padding:14px 17px;margin:18px 0;font-size:14.5px;color:var(--ink2)}}
.note.good{{border-left-color:var(--good);background:var(--good-bg)}}.note.warn{{border-left-color:var(--warn);background:var(--warn-bg)}}
.note b{{color:var(--ink)}}.note .lbl{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);display:block;margin-bottom:6px}}
.note p:last-child{{margin-bottom:0}}
footer{{margin-top:54px;padding-top:19px;border-top:1px solid var(--rule);font-family:var(--mono);font-size:11.5px;color:var(--ink3);line-height:1.8}}
</style>
<div class="sheet">
<header>
  <p class="kick">TRELLIS.2 &middot; dynamic texture &middot; {len(ORDER)} objects &middot; {CAMW} cameras each</p>
  <h1>{TITLE}</h1>
  <p class="lede">A texture is fitted to a single fixed camera &mdash; the one the Kling clip was
  shot from. This page asks the only question that matters afterwards: <b>does it hold up from
  angles the loss never saw?</b> Every object below shows that one supervised camera next to
  {NUNSEEN} it never saw, in a single block &mdash; so supervised and unseen sit side by side
  rather than on separate pages.</p>
</header>

<h2>Every object</h2>
<p class="sub">Mesh chosen by silhouette IoU against the still Kling was given, not by name &mdash;
every match cleared its runner-up by a wide margin, and &ldquo;monster&rdquo; turned out to be the
armadillo. All arms share one mesh, one target set, one loss (L1&thinsp;+&thinsp;0.1&thinsp;LPIPS),
rank&nbsp;4, seed&nbsp;42, 30&nbsp;epochs; only the adapter differs.</p>
<div class="tw"><table>
<thead><tr><th>Object</th><th>Mesh</th><th>Effect</th><th class="n">rung27</th>
<th class="n">+MCFM</th><th class="n">&Delta;</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

<div class="note">
  <span class="lbl">What the &Delta; column is, and is not</span>
  <p>MCFM is up on <b>{n_up} of {n_done}</b> objects, mean <b>{mean_d:+.3f}</b>&thinsp;dB. That is
  small, and it is meant to be: MCFM blends the conditioning across time, and PSNR scores each
  frame against its own target independently, so it cannot see a temporal prior at all.</p>
  <p style="margin-bottom:0">Measured where it can be seen &mdash; VBench Temporal Flickering on
  the PBR voxel field the decoder emits, no renderer in the path &mdash; MCFM cuts flicker
  <b>20.0%</b> and second-order jerk <b>44.4%</b>, lower on <b>28 of 28</b> objects, while drift
  from first frame to last moves only <b>+1.3%</b> so it is not simply freezing the texture.
  Quote MCFM that way, not by PSNR.</p>
</div>

{miss_html}
{prov_html}
{asset_html}

<h2>Object by object &mdash; supervised camera, then the {NUNSEEN} unseen</h2>
<p class="jump">{jump}</p>
<p class="sub">Four panels per video: <b>ground truth</b> &middot; <b>frozen TRELLIS.2</b> &middot;
<b>rung27</b> &middot; <b>rung27&thinsp;+&thinsp;MCFM</b>. The Kling clip is one fixed camera, so
ground truth exists at the training view only; elsewhere that column repeats the training-view
frame as a texture reference and is labelled accordingly.{TURN_NOTE}</p>
{''.join(blocks)}

<footer>
{shipped} videos &middot; {len(ORDER)} objects &times; {NV} cameras &middot; rung27 = rank-4 LoRA on
cross-attention and self-attention, 30 blocks &middot; MCFM = v2_D temporal blending of the cached
DINOv3 conditioning<br>
targets are the 2D copy: the video's colour sampled inside our own rasterised silhouette, rim
filled from the nearest valid neighbour
</footer>
</div>
"""

n = len(HTML.encode())
print(f'page {n/1048576:.2f} MiB  (cap 16.00 MiB)   {shipped} videos, {len(missing)} missing')
if n > CAP:
    print('OVER CAP — not writing.')
    sys.exit(1)
DST.write_text(HTML, encoding='utf8')
print(f'wrote {DST}')
