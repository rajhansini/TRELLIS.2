"""Supplementary page, rebuilt to the reviewer's guidelines.

THREE RULES, taken from the note and the reference page, not invented here:
  1. ONE METHOD PER VIDEO. The renders are [frozen | ours] composites; they are split
     so each half is its own clip. No video shows two methods.
  2. GALLERY FIGURES SHOW OURS ONLY. Frozen TRELLIS.2 is not in the paper's gallery,
     so putting it there splits the figure's message. Frozen appears ONLY in the
     figures that are actually comparisons (6, 8, 9), and there as its own clip.
  3. ONLY THE VIEWS THE PAPER PRESENTS. The paper's gallery shows two novel
     viewpoints, not four, so two are shown.
"""
import base64, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from supp_page import render

SPLIT = Path('/tmp/split')
B = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/supp_bundle')
have = {p.name for p in SPLIT.glob('*.mp4')}

def ours(sid, obj, view):
    n = f'{sid}_{obj}_{view}__ours.mp4'
    return n if n in have else None
def frozen(sid, obj, view):
    n = f'{sid}_{obj}_{view}__frozen.mp4'
    return n if n in have else None
def ref(sid, obj):
    n = f'{sid}_{obj}_reference.mp4'
    return n if n in have else None

def novel_views(sid, obj):
    """The two novel viewpoints, in whichever naming this object was rendered under."""
    for pair in (('y90','y270'), ('diagA','diagB')):
        if ours(sid, obj, pair[0]) and ours(sid, obj, pair[1]):
            return pair
    return None

def gallery_scene(sid, obj, prompt, name):
    """Reference + our result at the two novel views. OURS ONLY."""
    vs = novel_views(sid, obj)
    rows, cols = [], []
    r = ref(sid, obj)
    if r:
        rows.append(('reference video', [r]))
        cols = ['driving video']
    if vs:
        rows.append(('DynaMesh (ours)', [ours(sid,obj,vs[0]), ours(sid,obj,vs[1])]))
        cols = ['novel view 1', 'novel view 2']
        if r:  # reference occupies one column; pad so the grid stays aligned
            rows[0] = ('reference video', [r, None])
    return dict(name=name, prompt=prompt, cols=cols, rows=rows)

FIGS = []

FIGS.append(dict(id='fig1', num='1', head='DynaMesh carries a video’s evolving effect onto a 3D object.',
  intro='Reference video and our output. One method per clip.',
  caption='Given a reference video of the prompted effect unfolding on a render of the object, we '
          'produce a single mesh whose texture follows the video frame by frame. The geometry stays '
          'fixed while the appearance evolves.',
  scenes=[gallery_scene('fig1','plane_waves','Continuous, Hokusai waves gushing','aircraft')]))

FIGS.append(dict(id='fig2', num='2', head='Gallery of results.',
  intro='Four objects, four effects. Each shows the driving video and our output at the two novel '
        'viewpoints the paper presents. Frozen TRELLIS.2 is not part of this figure.',
  caption='Four objects driven by four different effects. The texture remains coherent even at '
          'viewpoints unseen in training.',
  scenes=[gallery_scene('fig2_1','hand_rorschach','Rorschach pattern forming','hand'),
          gallery_scene('fig2_2','chair_real_wooden_crack','Blue paint cracks and peels','chair'),
          gallery_scene('fig2_3','pumpkin_rot','Rot spreads and darkens','pumpkin'),
          gallery_scene('fig2_4','airplane_red_cracks','Red cracks spreading','airplane')]))

gen = sorted({n[len('fig5_lava_'):-len('__ours.mp4')] for n in have
              if n.startswith('fig5_lava_') and n.endswith('__ours.mp4')})
if gen:
    FIGS.append(dict(id='fig5', num='5', head='Generalization.',
      intro='Adapters fit on one reference video, applied without retraining to unseen meshes. '
            'Our output only.',
      caption='Per-scene adapters fit on a reference video are applied, without any retraining, to '
              'meshes never seen during fitting. Each object keeps its own geometry.',
      scenes=[dict(name='adapters transferred to unseen meshes', prompt=None,
                   cols=[t.replace('_',' ') for t in gen[:6]],
                   rows=[('rorschach', [f'fig5_ror_{t}__ours.mp4' if f'fig5_ror_{t}__ours.mp4' in have else None for t in gen[:6]]),
                         ('lava',      [f'fig5_lava_{t}__ours.mp4' for t in gen[:6]])])]))

# Figures 6, 8, 9 ARE comparisons -- frozen belongs here, as its own clip per row.
def compare_scene(sid, obj, prompt, name):
    vs = novel_views(sid, obj)
    if not vs: return None
    return dict(name=name, prompt=prompt, cols=['novel view 1','novel view 2'],
                rows=[('frozen TRELLIS.2', [frozen(sid,obj,vs[0]), frozen(sid,obj,vs[1])]),
                      ('DynaMesh (ours)',  [ours(sid,obj,vs[0]),   ours(sid,obj,vs[1])])])

for sid,num,obj,prompt,head,intro,cap in [
  ('fig6','6','chair_moss','Moss spreads and darkens','Flicker.',
   'A comparison figure, so both methods appear — each in its own clip, never combined.',
   'Frozen TRELLIS.2 changes the texture abruptly from one moment to the next, while our method '
   'keeps the moss growing smoothly, as in the video.'),
  ('fig8','8','spot_lava','Lava effect with cracks glowing in bright orange','Qualitative comparison.',
   'A comparison figure. One method per clip.',
   'The baselines either flicker or degrade the geometry at unseen views. Only our method tracks '
   'the effect consistently and preserves the shape’s geometry.'),
  ('fig9','9','vase_floral','Cobalt pattern drawing itself','Failure case.',
   'A comparison figure, shown as in the paper.',
   'DynaMesh relies on effects where the property of the pattern is local. Here it replicates the '
   'local patterns but does not recover them from unseen views, since the flowers and lines follow '
   'a global arrangement that a single view does not capture.')]:
    sc = compare_scene(sid, obj, prompt, obj.split('_')[0])
    if sc: FIGS.append(dict(id=sid, num=num, head=head, intro=intro, caption=cap, scenes=[sc]))

# Figure 7: the duck, at the two views the paper names. Ours only.
d1,d2 = 'fig7_bob_spots_d01__ours.mp4','fig7_bob_spots_sideH__ours.mp4'
if d1 in have:
    FIGS.append(dict(id='fig7', num='7', head='Making an existing static texture dynamic.',
      intro='Our output at the two views the paper shows. Frozen TRELLIS.2 is not part of this figure.',
      caption='Our method can operate on objects with a given texture and make it change over time. '
              'The dots glide around the ring while the head stays clear and the painted eyes, beak '
              'and molded shading hold their place, so only the pattern moves.',
      scenes=[dict(name='duck', prompt='Polka dots gliding over the duck',
                   cols=['first view','second view'],
                   rows=[('reference video',[ref('fig7','bob_spots'), None]),
                         ('DynaMesh (ours)',[d1, d2 if d2 in have else None])])]))
FIGS.sort(key=lambda f: int(f['num']))

SCOPE=('<b>Scope.</b> Sections follow the paper’s figures. Each video shows <b>one method</b>. '
       'Gallery figures (1, 2, 5, 7) show <b>our result only</b>, since frozen TRELLIS.2 is not '
       'part of those figures. Frozen appears only in the comparison figures (6, 8, 9), as its own clip.')

PREV=Path('/tmp/prevenc2'); PREV.mkdir(exist_ok=True)
import subprocess
used={}
def embed(n):
    if n in used: return used[n]
    small=PREV/n
    if not small.exists():
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(SPLIT/n),
                        '-vf','scale=520:-2:flags=lanczos,fps=12','-c:v','libx264','-crf','32',
                        '-preset','veryfast','-an',str(small)],check=False)
    used[n]='data:video/mp4;base64,'+base64.b64encode(small.read_bytes()).decode() if small.exists() else ''
    return used[n]

html=render(FIGS, SCOPE, embed)
Path('/net/projects/ranalab/rajhansini/TRELLIS.2/supp_preview.html').write_text(html,encoding='utf-8')
print(f'figures {len(FIGS)}  clips {len([v for v in used.values() if v])}  page {len(html.encode())/1e6:.1f} MB')
for f in FIGS:
    n=sum(1 for sc in f['scenes'] for _,cs in sc['rows'] for c in cs if c)
    meth=sorted({lab for sc in f['scenes'] for lab,_ in sc['rows']})
    print(f'  Fig {f["num"]}: {n} clips  rows={meth}')
