"""Build the self-contained GT-pipeline explainer artifact (images + video inlined)."""
import base64, os
from pathlib import Path

D = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
OUT = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/gt_explainer.html')


def b64(p, mime):
    return f'data:{mime};base64,' + base64.b64encode(Path(p).read_bytes()).decode()


VID = b64(D / 'gt_compare/original_vs_2Dcopy_vs_backproj.mp4', 'video/mp4')
IMG_ORIG = b64(T2 / 'data/teapot/frames_from_video/frame_0075.png', 'image/png')
IMG_2D = b64(D / 'gt_targets/frames/gt_0075.png', 'image/png')
IMG_BP = b64(D / 'gt_rendered_fix/frames/gt_0075.png', 'image/png')

# rung20 results — all from the tau=0.25 run, verified psnr=19.516
RUN = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/runs'
           '/rung20_all_kv_r4_s42_bbbce460')
IMG_SIDE = b64(RUN / 'diag/e030_gt_vs_ours.jpg', 'image/jpeg')
VID_360 = b64(D / 'rung20_tau0.25_1x360/rung20_kv_tau0.25_360_both_vs_frozen.mp4', 'video/mp4')
VID_720 = b64(D / 'rung20_tau0.25_2x360/rung20_kv_tau0.25_2x360_both_vs_frozen.mp4', 'video/mp4')

HTML = f'''<title>Building a Training Target — Three Pipelines</title>
<style>
  :root{{
    --paper:#FAF9F6; --card:#F2F1EC; --card-2:#E9E8E1;
    --ink:#17181C; --ink-2:#414450; --ink-3:#767A87;
    --rule:#E0DFD7; --rule-2:#C9C8BE;
    --flat:#0F6E72; --flat-bg:#E4F0F0;      /* 2D / screen-space  = cool */
    --surf:#A9541F; --surf-bg:#F7EAE0;      /* 3D / surface       = warm */
    --neut:#5B5F6B; --neut-bg:#ECECEA;
    --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }}
  @media (prefers-color-scheme:dark){{
    :root:not([data-theme="light"]){{
      --paper:#131519; --card:#1B1D23; --card-2:#22252C;
      --ink:#ECEDF0; --ink-2:#B9BCC6; --ink-3:#878B96;
      --rule:#2A2D35; --rule-2:#3A3E48;
      --flat:#5FC7CC; --flat-bg:#11292B;
      --surf:#E29668; --surf-bg:#2B1D14;
      --neut:#A0A5B2; --neut-bg:#212329;
    }}
  }}
  :root[data-theme="dark"]{{
    --paper:#131519; --card:#1B1D23; --card-2:#22252C;
    --ink:#ECEDF0; --ink-2:#B9BCC6; --ink-3:#878B96;
    --rule:#2A2D35; --rule-2:#3A3E48;
    --flat:#5FC7CC; --flat-bg:#11292B;
    --surf:#E29668; --surf-bg:#2B1D14;
    --neut:#A0A5B2; --neut-bg:#212329;
  }}
  *{{box-sizing:border-box}}
  body{{background:var(--paper); color:var(--ink); font-family:var(--sans);
    font-size:16.5px; line-height:1.62; margin:0; -webkit-font-smoothing:antialiased}}
  .sheet{{max-width:1080px; margin:0 auto; padding:0 28px 96px}}
  .col{{max-width:64ch}}
  header{{padding:60px 0 26px; border-bottom:2px solid var(--ink); margin-bottom:36px}}
  .kick{{font-family:var(--mono); font-size:11px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--ink-3); margin:0 0 16px}}
  h1{{font-size:clamp(27px,3.8vw,38px); font-weight:750; letter-spacing:-.02em;
    line-height:1.14; margin:0 0 14px; text-wrap:balance; max-width:20ch}}
  .lede{{font-size:17.5px; color:var(--ink-2); margin:0; max-width:62ch}}
  h2{{font-size:21px; font-weight:750; letter-spacing:-.012em; margin:48px 0 12px; text-wrap:balance}}
  h3{{font-size:16px; font-weight:700; margin:26px 0 8px}}
  p{{margin:0 0 14px}}
  code{{font-family:var(--mono); font-size:.86em; background:var(--card-2);
    padding:.1em .36em; border-radius:3px}}
  .num{{font-family:var(--mono); font-variant-numeric:tabular-nums}}

  .cards{{display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:18px; margin:26px 0}}
  .card{{background:var(--card); border:1px solid var(--rule); border-radius:9px;
    padding:20px 20px 18px; border-top:3px solid var(--rule-2)}}
  .card.flat{{border-top-color:var(--flat)}}
  .card.surf{{border-top-color:var(--surf)}}
  .card.neut{{border-top-color:var(--neut)}}
  .tag{{font-family:var(--mono); font-size:10.5px; letter-spacing:.13em;
    text-transform:uppercase; display:inline-block; padding:3px 8px; border-radius:4px; margin-bottom:11px}}
  .flat .tag{{background:var(--flat-bg); color:var(--flat)}}
  .surf .tag{{background:var(--surf-bg); color:var(--surf)}}
  .neut .tag{{background:var(--neut-bg); color:var(--neut)}}
  .card h3{{margin:0 0 4px; font-size:17px}}
  .card .sub{{color:var(--ink-3); font-size:13.5px; margin:0 0 14px; font-family:var(--mono)}}
  ol.steps{{margin:0; padding:0; list-style:none; counter-reset:s}}
  ol.steps li{{counter-increment:s; position:relative; padding-left:30px; margin-bottom:11px;
    font-size:15px; line-height:1.5}}
  ol.steps li::before{{content:counter(s); position:absolute; left:0; top:.05em;
    font-family:var(--mono); font-size:11px; width:20px; height:20px; border-radius:50%;
    display:grid; place-items:center; color:var(--paper)}}
  .flat ol.steps li::before{{background:var(--flat)}}
  .surf ol.steps li::before{{background:var(--surf)}}
  .neut ol.steps li::before{{background:var(--neut)}}
  .verdict{{margin-top:14px; padding-top:12px; border-top:1px solid var(--rule);
    font-size:14px; color:var(--ink-2)}}
  .verdict b{{color:var(--ink)}}

  figure{{margin:28px 0}}
  .stills{{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px}}
  .still{{background:var(--card); border:1px solid var(--rule); border-radius:8px; overflow:hidden}}
  .still img{{display:block; width:100%; height:auto; background:#fff}}
  .still .cap{{padding:9px 12px; font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
    border-top:1px solid var(--rule)}}
  video{{width:100%; height:auto; display:block; border:1px solid var(--rule); border-radius:8px; background:#fff}}
  figcaption{{font-size:14px; color:var(--ink-2); margin-top:11px; max-width:66ch; line-height:1.55}}

  .tablewrap{{overflow-x:auto; margin:22px 0; border:1px solid var(--rule); border-radius:8px}}
  table{{border-collapse:collapse; width:100%; font-size:14px; background:var(--card)}}
  th{{text-align:left; font-size:11px; letter-spacing:.07em; text-transform:uppercase;
    color:var(--ink-3); padding:10px 14px; border-bottom:1px solid var(--rule-2); white-space:nowrap}}
  td{{padding:10px 14px; border-bottom:1px solid var(--rule); color:var(--ink-2)}}
  tbody tr:last-child td{{border-bottom:none}}
  td:first-child{{color:var(--ink); font-weight:650}}
  td.n{{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap}}

  .callout{{background:var(--card); border-left:3px solid var(--ink-3); border-radius:0 7px 7px 0;
    padding:15px 19px; margin:22px 0}}
  .callout.key{{border-left-color:var(--flat)}}
  .callout.warn{{border-left-color:var(--surf)}}
  .callout p:last-child{{margin-bottom:0}}
  .callout .lbl{{font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
    text-transform:uppercase; color:var(--ink-3); display:block; margin-bottom:6px}}
  footer{{margin-top:56px; padding-top:20px; border-top:1px solid var(--rule);
    font-family:var(--mono); font-size:11.5px; color:var(--ink-3); line-height:1.7}}
</style>

<div class="sheet">

<header>
  <p class="kick">Dynamic texture · ground-truth pipeline</p>
  <h1>Three ways to build a training target</h1>
  <p class="lede">We have a video of a texture evolving and a fixed mesh. Before anything can be
  trained, each video frame has to become a <em>target image</em> the model's render can be
  compared against. There are three candidates, and the choice matters more than it looks.</p>
</header>

<section class="col">
  <h2>The problem</h2>
  <p>One camera, bolted in place. 150 frames. The object never moves — only its texture changes.</p>
  <p>The obvious move is to compare our render against the video frame directly. That fails,
  because the two do not have the same outline:</p>
</section>

<div class="tablewrap col">
  <table>
    <tbody>
      <tr><td>video's teapot</td><td class="n">102,549 → 109,583 px</td><td>grows over the sequence</td></tr>
      <tr><td>our render</td><td class="n">28,559 px</td><td>constant — mesh and camera are fixed</td></tr>
    </tbody>
  </table>
</div>

<section class="col">
  <p>The video came from a generative model, and it reshaped the object. So every frame has a
  band where one image has surface and the other has background. A pixel-by-pixel loss would
  charge the adapter for that <em>shape</em> difference as though it were texture error.</p>
  <div class="callout key">
    <span class="lbl">What every solution must do</span>
    <p>Produce a target that has <b>our mesh's outline</b>, so the loss compares texture and never shape.</p>
  </div>
</section>

<h2>The three pipelines</h2>

<div class="cards">

  <div class="card neut">
    <span class="tag">Option A</span>
    <h3>Raw video frame</h3>
    <p class="sub">960 × 960 · use as-is</p>
    <ol class="steps">
      <li>Take the frame straight from the video model.</li>
      <li>Compare it against our render.</li>
    </ol>
    <div class="verdict"><b>Rejected.</b> Different outline, different size. The loss would be
    part shape, part texture, and the shape part grows over the sequence.</div>
  </div>

  <div class="card flat">
    <span class="tag">Option B — in use</span>
    <h3>2D copy</h3>
    <p class="sub">screen space · no 3D</p>
    <ol class="steps">
      <li><b>Render the mesh once</b> to get a mask: which pixels our object occupies.</li>
      <li><b>Find the object</b> in the video frame by threshold, keeping the largest blob.</li>
      <li><b>Flood its colour outward</b> over the white background.</li>
      <li><b>Shrink</b> 960 → 518, averaging colours, nearest-neighbour for the mask.</li>
      <li><b>Copy into our mask</b> — at each pixel we occupy, take the video's colour at that
      same pixel. Leave everything else white.</li>
    </ol>
    <div class="verdict"><b>Why pixel p works:</b> the camera never moves and the object never
    moves, so pixel p looks at the same spot in both images. <b>100% coverage.</b></div>
  </div>

  <div class="card surf">
    <span class="tag">Option C</span>
    <h3>Backprojection</h3>
    <p class="sub">through the 3D surface</p>
    <ol class="steps">
      <li><b>Rasterise the mesh.</b> Each pixel learns which triangle it sees and where inside it.</li>
      <li><b>Hand each pixel's colour to that triangle's three corners</b>, weighted by how close
      it landed to each.</li>
      <li><b>Average per vertex</b> across every contributing pixel. Colour now lives on the mesh.</li>
      <li><b>Re-render</b> those vertex colours through the same mesh and camera.</li>
    </ol>
    <div class="verdict"><b>Stronger claim:</b> colour is attached to a surface point, not assumed
    by pixel position. <b>~99% coverage</b> — see below.</div>
  </div>

</div>

<section class="col">
  <div class="callout">
    <span class="lbl">Step 3 of Option B, the one people ask about</span>
    <p>Why flood the colour over the background before shrinking? Because shrinking averages
    neighbouring pixels. At the object's edge you would be averaging teapot with white
    background, producing a bright halo that exists in neither image — and that halo then gets
    copied inward as if it were texture. Remove the white first and there is nothing bright to
    average with.</p>
  </div>
</section>

<h2>Same frame, all three</h2>

<figure>
  <div class="stills">
    <div class="still"><img src="{IMG_ORIG}" alt="Original video frame 75">
      <div class="cap">A · original video · 960²</div></div>
    <div class="still"><img src="{IMG_2D}" alt="2D copy target frame 75">
      <div class="cap">B · 2D copy · 518²</div></div>
    <div class="still"><img src="{IMG_BP}" alt="Backprojected target frame 75">
      <div class="cap">C · backprojected · 518²</div></div>
  </div>
  <figcaption>Frame 75. <b>A</b> has the video model's outline; <b>B</b> and <b>C</b> both carry
  our mesh's outline, which is the point of doing either. The difference between B and C is a
  thin unfilled fringe along the silhouette in C.</figcaption>
</figure>

<h2>The comparison, all 150 frames</h2>

<figure>
  <video controls loop muted playsinline preload="metadata" src="{VID}"></video>
  <figcaption>Left to right: original video · 2D copy · backprojected. Watch the outline —
  the left panel's object grows through the sequence while the two targets stay pinned to our
  mesh. That is the shape mismatch being removed.</figcaption>
</figure>

<h2>How B and C differ</h2>

<div class="tablewrap">
  <table>
    <thead><tr><th></th><th class="n">frame 1</th><th class="n">frame 75</th><th class="n">frame 150</th></tr></thead>
    <tbody>
      <tr><td>C: pixels with no colour</td><td class="n">543</td><td class="n">320</td><td class="n">233</td></tr>
      <tr><td>as % of silhouette</td><td class="n">1.90%</td><td class="n">1.12%</td><td class="n">0.82%</td></tr>
      <tr><td>agreement between B and C</td><td class="n">31.3 dB</td><td class="n">28.8 dB</td><td class="n">27.7 dB</td></tr>
    </tbody>
  </table>
</div>

<section class="col">
  <p>The two agree closely. C's only visible flaw is that thin fringe, and it <em>shrinks</em>
  over the sequence.</p>

  <h3>Why C leaves a fringe at all</h3>
  <p>Each pixel's colour comes from its triangle's three corners. But the mesh has
  <span class="num">430,910</span> faces while the render resolves only about
  <span class="num">98,000</span> pixels — roughly <b>4.4× denser than the camera can see</b>.
  So <span class="num">173,951</span> of <span class="num">215,462</span> vertices are never hit
  by any pixel, in any frame. A triangle with no supervised corner has no colour to give, and
  re-rendering leaves a gap.</p>
  <p>Normalising by how much of each blend came from supervised corners recovered about half the
  fringe. It could not do more: redistributing signal cannot create signal.</p>

  <h3>Why B was chosen</h3>
  <p>Not because C is broken — it isn't. Because B is simpler and has no ceiling: 0% unfilled
  instead of ~1%, one script instead of two, and no dependence on how dense the mesh happens to
  be, so it transfers to any object.</p>

  <div class="callout warn">
    <span class="lbl">The honest limitation of B</span>
    <p>It assumes the video's object sits where our mesh sits. Where our outline extends past the
    video's, there is no colour to copy and step 3's flood fills in. For the teapot that is
    <b>319 px, 1.1%</b> — invisible. On a second object where our mesh rendered larger than the
    video's, it was <b>17,613 px, 36%</b>, and produced visible streaking. The assumption is
    cheap to check per object and should be checked before training.</p>
  </div>

  <div class="callout">
    <span class="lbl">What going through 3D does and doesn't buy</span>
    <p>C attaches colour to a surface point rather than assuming it from pixel position — a
    genuinely stronger claim. But it does <b>not</b> fix a size mismatch either: projecting our
    mesh still samples the video wherever it lands. Both methods depend on the object occupying
    the same screen region, and neither recovers it when the video model changes the shape.</p>
    <p>The round trip earns its cost the moment the camera <em>moves</em>. At a fixed viewpoint
    there is no pixel correspondence to gain — only precision to lose.</p>
  </div>
</section>

<h2>rung20 — what we do with the target</h2>

<section class="col">
  <p>The targets above are the input to training. Everything below runs on top of them.</p>
</section>

<div class="cards">
  <div class="card flat">
    <span class="tag">Set up once</span>
    <h3>Before training</h3>
    <p class="sub">nothing here changes during the run</p>
    <ol class="steps">
      <li><b>Freeze the whole generator.</b> All 4B parameters of TRELLIS.2, untouched.</li>
      <li><b>Attach a small adapter</b> where the image enters the 3D — the cross-attention layers,
      in all 30 blocks. <b>491,520 parameters, 0.038%</b> of the generator.</li>
      <li><b>Build the 150 targets</b> — the 2D copy from the previous section.</li>
      <li><b>Build a trust map</b>: how much to believe each pixel, from how head-on the surface is
      and how much surface that pixel covers. Below a cutoff, a pixel is dropped entirely —
      <b>27,088 of 28,559 kept</b>.</li>
    </ol>
    <div class="verdict"><b>Why cross-attention:</b> shape reaches the generator one way, the image
    another. Adapting the image path changes what the picture is allowed to say, while leaving
    frozen the machinery that spreads it across the object.</div>
  </div>

  <div class="card surf">
    <span class="tag">Repeat · 30 epochs × 150 frames</span>
    <h3>One training step</h3>
    <p class="sub">one frame at a time</p>
    <ol class="steps">
      <li><b>Pick a frame</b>, and a random point along the 12-step generation trajectory.</li>
      <li><b>Run the generator up to that point</b> with gradients off — reaching a state the real
      sampler would actually pass through.</li>
      <li><b>Take one step with gradients on.</b> Only one: storing all twelve would not fit.</li>
      <li><b>Decode</b> to a colour for every voxel, then <b>render</b> it through the mesh.</li>
      <li><b>Compare to the target</b>, each pixel weighted by the trust map.</li>
      <li><b>Update the adapter only.</b> The generator never changes.</li>
    </ol>
    <div class="verdict"><b>Result:</b> frozen TRELLIS.2 scores <b>9.60 dB</b> against the target.
    After training, <b>19.51 dB</b>.</div>
  </div>
</div>

<section class="col">
  <div class="callout key">
    <span class="lbl">The bet the whole design rests on</span>
    <p>The camera resolves only about <b>17% of the object</b>. Nothing supervises the rest — not
    the back, not the underside. The adapter is placed at cross-attention specifically so that the
    frozen self-attention downstream carries the edit outward to the parts no pixel ever sees.
    The turntable videos below are what test that, not the score.</p>
  </div>
</section>

<h2>Results</h2>

<figure>
  <img src="{IMG_SIDE}" alt="Training view: target beside model output at three points in the sequence"
       style="width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:8px;background:#fff">
  <figcaption><b>The training view.</b> Target on the left of each pair, model output on the right,
  at three points through the sequence. This is the one angle the model was ever shown — it should
  look right here, and it does.</figcaption>
</figure>

<figure>
  <video controls loop muted playsinline preload="metadata" src="{VID_360}"></video>
  <figcaption><b>One full turn.</b> Frozen TRELLIS.2 on the left, adapted on the right, while the
  texture advances through all 150 frames. Everything away from the front is unsupervised —
  it is there because the frozen self-attention propagated the edit, not because anything taught it.</figcaption>
</figure>

<figure>
  <video controls loop muted playsinline preload="metadata" src="{VID_720}"></video>
  <figcaption><b>Two full turns.</b> The same 150 frames, camera going round twice, so every point on
  the surface is seen <em>twice at different moments</em> in the texture's evolution. This is the
  test that separates a texture living on the surface from one painted on at a fixed angle —
  if it were painted on, the second pass would not agree with the first.</figcaption>
</figure>

<section class="col">
  <div class="callout warn">
    <span class="lbl">What is still open</span>
    <p>Bright patches remain along the silhouette rim. The trust map discards grazing pixels, so
    those regions end up with no supervision and no prior holding them — and the amount discarded
    has never been tuned against a measurement, only judged by eye. Two settings were tried:
    keeping 94.8% of pixels gives <b>19.51 dB</b>, keeping 85.4% gives <b>17.96 dB</b>. Using every
    pixel with no trust map at all gives <b>22.30 dB</b> — but more rim artefacts. That trade is
    the open question.</p>
  </div>
</section>

<footer>
  A · data/teapot/frames_from_video/frame_*.png &nbsp;·&nbsp; 960², straight from the video model<br>
  B · out/gt_targets/frames/gt_*.png &nbsp;·&nbsp; make_gt_targets.py &nbsp;·&nbsp; what rung17–24 train on<br>
  C · out/gt_rendered_fix/frames/gt_*.png &nbsp;·&nbsp; backproject_gt.py → render_backproj_gt.py<br>
  All three: 150 frames, one fixed camera, mesh 215,462 verts / 430,910 faces, render 518².<br>
  rung20 results · runs/rung20_all_kv_r4_s42_bbbce460 · tau 0.25 · rank 4 · 30 epochs · verified psnr 19.516
</footer>

</div>
'''

OUT.write_text(HTML)
print(f'[WROTE] {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)')
