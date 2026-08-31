"""Build both outputs from the shared matrix generator: the shipped bundle and the
embedded-video preview. Same figure list, same layout, so they cannot drift."""
import base64, json, os, sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from supp_page import render

B = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/supp_bundle')
V = B / 'videos'
have = {p.name for p in V.glob('*.mp4')}


def pick(*cands):
    for c in cands:
        if c in have:
            return c
    return None


VIEWS4 = ['supervised', 'unseen A', 'unseen B', 'unseen C']
YAW4 = ['0°', '90°', '180°', '270°']


def row_yaw(obj, sid):
    return [pick(f'{sid}_{obj}_y0'  + '.mp4'), pick(f'{sid}_{obj}_y90' + '.mp4'),
            pick(f'{sid}_{obj}_y180'+ '.mp4'), pick(f'{sid}_{obj}_y270'+ '.mp4')]


def row_diag(obj, sid):
    return [pick(f'{sid}_{obj}_train.mp4'), pick(f'{sid}_{obj}_diagA.mp4'),
            pick(f'{sid}_{obj}_diagB.mp4'), pick(f'{sid}_{obj}_diagC.mp4')]


def scene(obj, sid, prompt, name):
    """One object: a reference row (single clip) and a 4-view result row."""
    yaw = row_yaw(obj, sid)
    cols, cells = (YAW4, yaw) if any(yaw) else (VIEWS4, row_diag(obj, sid))
    ref = pick(f'{sid}_{obj}_reference.mp4')
    rows = []
    if ref:
        rows.append(('reference', [ref] + [None]*(len(cols)-1)))
    rows.append(('frozen | ours', cells))
    return dict(name=name, prompt=prompt, cols=cols, rows=rows)


FIGS = [
 dict(id='fig1', num='1', head='DynaMesh carries a video’s evolving effect onto a 3D object.',
      intro='The reference video shows the prompted effect unfolding on a render of the object; '
            'our output follows it frame by frame while the geometry stays fixed.',
      caption='Given a reference video of the prompted effect unfolding on a render of the '
              'object, we produce a single mesh whose texture follows the video frame by frame. '
              'The geometry stays fixed while the appearance evolves, so every moment of the '
              'sequence can be re-rendered from any viewpoint.',
      scenes=[scene('plane_waves', 'fig1', 'Continuous, Hokusai waves gushing', 'aircraft')]),
 dict(id='fig2', num='2', head='Gallery of results.',
      intro='Four objects driven by four different effects, each shown at the supervised view '
            'and three views never supervised.',
      caption='Four objects driven by four different effects. The texture remains coherent even '
              'at viewpoints unseen in training.',
      scenes=[scene('hand_rorschach','fig2_1','Rorschach pattern forming','hand'),
              scene('chair_real_wooden_crack','fig2_2','Blue paint cracks and peels','chair'),
              scene('pumpkin_rot','fig2_3','Rot spreads and darkens','pumpkin'),
              scene('airplane_red_cracks','fig2_4','Red cracks spreading','airplane')]),
 dict(id='fig6', num='6', head='Flicker.',
      intro='The section that cannot exist on paper: flicker is a difference between consecutive '
            'frames, so no still can show it.',
      caption='A chair with moss spreading over its surface. Frozen TRELLIS.2 (left half of each '
              'clip) changes the texture abruptly from one moment to the next, while our method '
              'keeps the moss growing smoothly, as in the video.',
      scenes=[scene('chair_moss','fig6','Moss spreads and darkens','chair')]),
 dict(id='fig7', num='7', head='Making an existing static texture dynamic.',
      intro='An object that already carries its own texture; the prompted effect sets it in motion.',
      caption='Our method can operate on objects with a given texture and make it change over '
              'time. The dots glide around the ring while the head stays clear and the painted '
              'eyes, beak and molded shading hold their place, so only the pattern moves.',
      scenes=[dict(name='duck', prompt='Polka dots gliding over the duck',
                   cols=['first view','second view'],
                   rows=[('reference',[pick('fig7_bob_spots_reference.mp4'),None]),
                         ('frozen | ours',[pick('fig7_bob_spots_d01.mp4'),
                                           pick('fig7_bob_spots_sideH.mp4')])])]),
 dict(id='fig8', num='8', head='Qualitative comparison.',
      intro='Our result against the reference on the video used in the paper; the baseline rows '
            'appear in the PDF.',
      caption='The baselines either flicker or degrade the geometry at unseen views. Only our '
              'method tracks the effect consistently and preserves the shape’s geometry.',
      scenes=[scene('spot_lava','fig8','Lava effect with cracks glowing in bright orange','cow')]),
 dict(id='fig9', num='9', head='Failure case.',
      intro='Included deliberately: the limitation reads more honestly in motion than from a still.',
      caption='DynaMesh relies on effects where the property of the pattern is local. Here it '
              'replicates the local patterns, but does not recover them from unseen views, since '
              'the flowers and lines follow a global arrangement that a single view does not capture.',
      scenes=[scene('vase_floral','fig9','Cobalt pattern drawing itself','vase')]),
]

# Figure 5 -- a true matrix: rows are the two adapters, columns are unseen target meshes.
gen_targets = sorted({n[len('fig5_lava_'):-4] for n in have if n.startswith('fig5_lava_')})
if gen_targets:
    show = gen_targets            # ALL targets: the bundle must reference every clip
    FIGS.insert(2, dict(
        id='fig5', num='5', head='Generalization.',
        intro='Per-scene adapters fit on one reference video, applied without retraining to '
              'meshes never seen during fitting. Columns are the unseen target meshes.',
        caption='Per-scene adapters fit on a reference video are applied, without any retraining, '
                'to meshes never seen during fitting. Both effects clearly propagate on the new '
                'shapes according to the video. The adapters change only how appearance is '
                'generated, and each object keeps its own geometry.',
        scenes=[dict(name='adapters transferred to unseen meshes', prompt=None,
                     cols=[t.replace('_',' ') for t in show],
                     rows=[('rorschach',[pick(f'fig5_ror_{t}.mp4') for t in show]),
                           ('lava',     [pick(f'fig5_lava_{t}.mp4') for t in show])])]))

SCOPE = ('<b>Scope.</b> Sections correspond to the figures of the main paper and appear in '
         'figure order. Columns are viewpoints; rows are what is shown. {extra}')

# ---- 1. the shipped bundle: plain relative srcs, every clip
html = render(FIGS, SCOPE.format(extra='All clips are local files in <code>videos/</code>; '
              'the page opens offline with no server.'), lambda n: f'videos/{n}')
(B / 'index.html').write_text(html, encoding='utf-8')

# ---- 2. the preview: the same page with clips embedded, capped for the artifact limit
# Every clip is embedded, from a PREVIEW-QUALITY re-encode (560px wide, 12 fps, crf 32)
# that is roughly eight times smaller than the shipped clip. The earlier budget-and-skip
# approach left the page mostly dashed "in full version" boxes, which made a style
# preview look broken -- the whole point is that a reader sees the finished layout.
PREV = Path('/tmp/prevenc')
used = {}
def embed(n):
    if n in used: return used[n]
    src = PREV / n
    if not src.exists():
        src = V / n
    used[n] = 'data:video/mp4;base64,' + base64.b64encode(src.read_bytes()).decode()
    return used[n]

prev = render(FIGS, SCOPE.format(extra='This preview embeds a subset of the clips; the submitted '
              'bundle carries all <b>65</b> and opens offline with no server.'), embed)
Path('/net/projects/ranalab/rajhansini/TRELLIS.2/supp_preview.html').write_text(prev, encoding='utf-8')
print(f'bundle index.html rebuilt ({len(html)/1e3:.0f} KB)')
print(f'preview: {len(used)} clips embedded, page {len(prev.encode())/1e6:.1f} MB')
