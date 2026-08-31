"""build_supp_bundle.py -- standalone supplementary HTML page + videos, zipped.

WHY A PAGE AND NOT MORE STILLS. The paper's claim is temporal: an effect that
progresses over time and stays coherent from viewpoints the video never showed. A
still cannot carry either. Every section below pairs the REFERENCE VIDEO with our
result at the viewpoints that figure presents, so a reviewer can check the claim
directly.

SECTIONS ARE THE PAPER'S FIGURE HEADLINES, verbatim, so the page maps one-to-one
onto the PDF.

THE PANEL VIDEOS ARE ALREADY [frozen | ours] COMPOSITES. That is deliberate and kept:
the comparison a reviewer most wants is exactly the one the composite makes, and it
removes any doubt that the two rows are the same camera and the same frames.
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
OUT = T2 / 'supp_bundle'
VID = OUT / 'videos'
for d in (OUT, VID):
    d.mkdir(parents=True, exist_ok=True)

YAWS = ['y0', 'y90', 'y180', 'y270']
YAWLBL = {'y0': '0°', 'y90': '90°', 'y180': '180°', 'y270': '270°'}


def ref_video(obj):
    c = sorted((T2 / 'data' / obj / 'video').glob('*.mp4'))
    return c[0] if c else None


def panel_video(dirname):
    d = E / 'out' / dirname
    c = sorted(d.glob('*.mp4'))
    return c[0] if c else None


def take(src, name):
    """Copy one mp4 into the bundle under a readable name. Returns the rel path."""
    if src is None or not Path(src).exists():
        return None
    dst = VID / f'{name}.mp4'
    if not dst.exists():
        shutil.copy2(src, dst)
    return f'videos/{name}.mp4'


# ---------------------------------------------------------------- the sections
# (id, paper headline, blurb, [ (row-label, [(caption, relpath)]) ])
SECTIONS = []


ALIAS = {'plane_waves': 'plane_waves_from_frame_150'}


def result_dirs(obj):
    obj = ALIAS.get(obj, obj)
    """Where an object's result videos live. Two naming families exist and BOTH are
    real: the yaw panels (panel_<obj>_mcfm_y0..y270) were built for the gallery
    figures, and the four fixed cameras (view_<obj>_27m_{train,diagA,diagB,diagC})
    for the metrics. Objects have one or the other, not always both, so trying only
    the first silently produced sections holding nothing but the reference video."""
    out = []
    for y in YAWS:
        d = E / 'out' / f'panel_{obj}_mcfm_{y}'
        if d.is_dir(): out.append((y, f'view {YAWLBL[y]}', d.name))
    if out: return out
    for v, lbl in [('train', 'supervised view'), ('diagA', 'unseen view A'),
                   ('diagB', 'unseen view B'), ('diagC', 'unseen view C')]:
        d = E / 'out' / f'view_{obj}_27m_{v}'
        if d.is_dir(): out.append((v, lbl, d.name))
    return out


def add_object_section(sid, headline, blurb, obj, prompt, dirpat=None):
    rows = []
    r = take(ref_video(obj), f'{sid}_{obj}_reference')
    if r:
        rows.append(('Reference video (input to our pipeline)', [(prompt, r)]))
    vids = []
    for key, lbl, dname in result_dirs(obj):
        p = take(panel_video(dname), f'{sid}_{obj}_{key}')
        if p:
            vids.append((lbl, p))
    if vids:
        rows.append(('Our result — frozen TRELLIS.2 (left) vs DynaMesh (right)', vids))
    if rows:
        SECTIONS.append((sid, headline, blurb, obj, rows))


# Fig 1 -- teaser
add_object_section(
    'fig1', 'Figure 1. DynaMesh carries a video’s evolving effect onto a 3D object as a texture that changes over time',
    'The reference video shows the prompted effect unfolding on a render of the object. '
    'Our output follows it frame by frame while the geometry stays fixed, so every moment '
    'can be re-rendered from any viewpoint.',
    'plane_waves', '“Continuous, Hokusai waves gushing”')

# Fig 2 -- gallery
GALLERY = [('hand_rorschach', '“Rorschach pattern forming”'),
           ('chair_real_wooden_crack', '“Blue paint cracks and peels”'),
           ('pumpkin_rot', '“Rot spreads and darkens”'),
           ('airplane_red_cracks', '“Red cracks spreading”')]
for i, (obj, prompt) in enumerate(GALLERY, 1):
    add_object_section(
        f'fig2_{i}', f'Figure 2. Gallery of results — {prompt}',
        'Four objects driven by four different effects. The texture remains coherent '
        'even at viewpoints unseen in training.',
        obj, prompt)

# Fig 5 -- generalization
gen_rows = []
for eff, label, prompt in [('ror', 'Rorschach', '“Rorschach effect with spots growing”'),
                           ('lava', 'Lava', '“Lava effect with cracks glowing in bright orange”')]:
    tgts = sorted({re.sub(r'_yaw\d+$', '', p.name).replace(f'r36_{eff}_', '')
                   for p in (E / 'out').glob(f'r36_{eff}_*_yaw000')})
    vids = []
    for t in tgts:
        p = take(panel_video(f'r36_{eff}_{t}_yaw000'), f'fig5_{eff}_{t}')
        if p:
            vids.append((t.replace('_', ' '), p))
    if vids:
        gen_rows.append((f'{label} adapter, {prompt}, applied to unseen meshes', vids))
if gen_rows:
    SECTIONS.append(('fig5', 'Figure 5. Generalization',
                     'Per-scene adapters fit on a reference video are applied, without any '
                     'retraining, to meshes never seen during fitting. Each object keeps its '
                     'own geometry; only how appearance is generated changes.',
                     None, gen_rows))

# Fig 6 -- flicker
add_object_section(
    'fig6', 'Figure 6. Flicker',
    'A chair with moss spreading over its surface. Frozen TRELLIS.2 (left half of each '
    'video) changes the texture abruptly from one moment to the next; DynaMesh (right half) '
    'keeps the moss growing smoothly, as in the video. This is the comparison a still cannot show.',
    'chair_moss', '“Moss spreads and darkens”')

# Fig 7 -- existing texture
rows7 = []
r = take(ref_video('bob_spots'), 'fig7_bob_spots_reference')
if r:
    rows7.append(('Reference video', [('“Polka dots gliding over the duck”', r)]))
v7 = []
for vn, lbl in [('d01', 'First view'), ('sideH', 'Second view')]:
    p = take(panel_video(f'view_bob_spots_27m_{vn}'), f'fig7_bob_spots_{vn}')
    if p:
        v7.append((lbl, p))
if v7:
    rows7.append(('Our result at the two views shown in the paper', v7))
if rows7:
    SECTIONS.append(('fig7', 'Figure 7. Making an existing static texture dynamic',
                     'The duck already carries its own texture; the prompted effect sets the '
                     'dots in motion while the head stays clear and the painted eyes, beak and '
                     'molded shading hold their place, so only the pattern moves.',
                     'bob_spots', rows7))

# Fig 8 -- qualitative comparison
rows8 = []
r = take(ref_video('spot_lava'), 'fig8_spot_lava_reference')
if r:
    rows8.append(('Reference video', [('“Lava effect with cracks glowing in bright orange”', r)]))
v8 = []
for key, lbl, dname in result_dirs('spot_lava'):
    p = take(panel_video(dname), f'fig8_spot_lava_{key}')
    if p:
        v8.append((lbl, p))
if v8:
    rows8.append(('DynaMesh (right half) vs frozen TRELLIS.2 (left half)', v8))
if rows8:
    SECTIONS.append(('fig8', 'Figure 8. Qualitative comparison',
                     'The baselines either flicker or degrade the geometry at unseen views. '
                     'The frozen generator reconstructs each frame independently; only our '
                     'method tracks the effect consistently while preserving the geometry. '
                     'Baseline clips are in the paper; shown here is the reference against '
                     'ours at every available camera.',
                     'spot_lava', rows8))

# Fig 9 -- failure case
add_object_section(
    'fig9', 'Figure 9. Failure case',
    'DynaMesh relies on effects whose pattern is LOCAL. Here the supervised view came out as '
    'intended, but the pattern loses detail at unseen views, because the effect is defined not '
    'only by the local flowers but by their global arrangement, which a single view does not capture. '
    'Included deliberately: the limitation is easier to judge in motion than from a still.',
    'vase_floral', '“Cobalt pattern drawing itself”')

# ---------------------------------------------------------------- the page
CSS = """
:root{--bg:#fbfbfc;--fg:#16181d;--mut:#5b6472;--line:#e3e6ec;--card:#fff;--accent:#2f5fd0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:38px 28px 26px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 6px;font-size:26px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:14px}
nav{padding:14px 28px;border-bottom:1px solid var(--line);background:var(--card);
 position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px}
nav a{font-size:12.5px;color:var(--accent);text-decoration:none;border:1px solid var(--line);
 padding:4px 9px;border-radius:999px;background:#fff;white-space:nowrap}
nav a:hover{background:#f0f4ff}
main{padding:8px 28px 64px;max-width:1500px;margin:0 auto}
section{margin:34px 0 0;padding:22px 0 0;border-top:1px solid var(--line)}
section:first-of-type{border-top:none}
h2{font-size:18px;margin:0 0 6px;letter-spacing:-.01em}
.blurb{color:var(--mut);font-size:13.5px;margin:0 0 16px;max-width:76ch}
.rowlabel{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
 margin:16px 0 8px;font-weight:600}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(310px,1fr))}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;
 overflow:hidden}
video{width:100%;display:block;background:#eef0f4}
figcaption{padding:8px 10px;font-size:12.5px;color:var(--mut);border-top:1px solid var(--line)}
footer{padding:26px 28px 60px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--line)}
@media (prefers-color-scheme:dark){
 :root{--bg:#0f1115;--fg:#e8eaee;--mut:#98a2b3;--line:#252a33;--card:#161a20;--accent:#7aa2f7}
 nav a{background:var(--card)} nav a:hover{background:#1d2430} video{background:#0b0d11}}
"""

html = ['<meta charset="utf-8"><title>DynaMesh — Supplementary Results</title>',
        f'<style>{CSS}</style>',
        '<header><h1>DynaMesh: Dynamic 3D Texture Generation</h1>',
        '<div class="sub">Supplementary video results &middot; 3DV 2027 Submission #92 &middot; '
        'sections follow the figures of the main paper</div></header>',
        '<nav>' + ''.join(f'<a href="#{s[0]}">{s[1].split(".")[0]}. {s[1].split("—")[0].split(".",1)[1].strip()[:34]}</a>'
                          for s in SECTIONS) + '</nav>', '<main>']
for sid, headline, blurb, obj, rows in SECTIONS:
    html.append(f'<section id="{sid}"><h2>{headline}</h2><p class="blurb">{blurb}</p>')
    for label, vids in rows:
        html.append(f'<div class="rowlabel">{label}</div><div class="grid">')
        for cap, rel in vids:
            html.append(f'<figure><video src="{rel}" controls loop muted playsinline preload="metadata">'
                        f'</video><figcaption>{cap}</figcaption></figure>')
        html.append('</div>')
    html.append('</section>')
html.append('</main><footer>All videos are 150 frames unless noted (pumpkin_rot is 121). '
            'Result videos are [frozen TRELLIS.2 | DynaMesh] composites sharing one camera '
            'and one frame index, except the reference videos, which are the raw model output. '
            'Anonymised for review.</footer>')
(OUT / 'index.html').write_text('\n'.join(html), encoding='utf-8')

nvid = len(list(VID.glob('*.mp4')))
size = sum(p.stat().st_size for p in VID.glob('*.mp4'))/1e6
print(f'sections: {len(SECTIONS)}   videos: {nvid}   {size:.1f} MB')
for sid, h, _, _, rows in SECTIONS:
    print(f'  {sid:8s} {sum(len(v) for _, v in rows):2d} videos  {h[:66]}')
