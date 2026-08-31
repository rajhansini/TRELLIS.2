"""supp_page.py -- one page generator for both the shipped bundle and the preview.

LAYOUT IS A LABELLED MATRIX, not a card grid. Columns are viewpoints, rows are what
is being shown, and the labels live once in the header row and the left rail instead
of being repeated as a caption under every clip. That is the convention academic
project pages use for method comparisons, and it is what makes a reader able to scan
DOWN a column to hold the viewpoint fixed and ACROSS a row to hold the method fixed.

Both outputs come from here so the preview cannot drift from the submitted page.
"""
import base64, os
from pathlib import Path

CSS = """
:root{--paper:#fff;--ink:#111317;--mut:#606671;--rule:#dcdfe4;--hair:#eceef1;
      --link:#1b3f7a;--well:#f7f8f9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0f1114;--ink:#e8eaed;--mut:#98a0aa;--rule:#282c33;--hair:#1c2026;
  --link:#8fb0e8;--well:#15181d}}
:root[data-theme="dark"]{--paper:#0f1114;--ink:#e8eaed;--mut:#98a0aa;--rule:#282c33;
  --hair:#1c2026;--link:#8fb0e8;--well:#15181d}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font:400 16px/1.6 "Source Serif 4",Georgia,"Times New Roman",serif}
.wrap{max-width:1120px;margin:0 auto;padding:0 30px}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}

.title{text-align:center;padding:58px 0 34px}
.venue{font:400 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;
 text-transform:uppercase;color:var(--mut);margin-bottom:20px}
h1{font:600 33px/1.2 "Source Serif 4",Georgia,serif;margin:0 0 9px;letter-spacing:-.01em;
 text-wrap:balance}
.byline{color:var(--mut);font-size:15px;margin:0}
.scope{margin:28px auto 0;max-width:68ch;text-align:left;font-size:14px;color:var(--mut);
 border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);padding:13px 0}
.scope b{color:var(--ink);font-weight:600}

.toc{padding:22px 0 30px;border-bottom:2px solid var(--rule)}
.toc h2{font:500 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;text-transform:uppercase;
 color:var(--mut);margin:0 0 11px}
.toc ol{margin:0;padding:0;list-style:none;display:grid;gap:5px 24px;
 grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.toc a{color:var(--link);text-decoration:none;font-size:15px}
.toc a:hover{text-decoration:underline}
.toc a:focus-visible{outline:2px solid var(--link);outline-offset:2px}
.toc .n{color:var(--mut);font:400 12.5px/1 "IBM Plex Mono",monospace;margin-right:8px}

section{padding:40px 0 12px;border-top:1px solid var(--rule)}
section:first-of-type{border-top:none}
.fignum{font:500 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.12em;
 text-transform:uppercase;color:var(--mut);margin-bottom:9px}
h2.head{font:600 23px/1.28 "Source Serif 4",Georgia,serif;margin:0 0 8px;
 letter-spacing:-.01em;text-wrap:balance}
.intro{color:var(--mut);font-size:14.5px;max-width:70ch;margin:0 0 26px}

.scene{margin:0 0 30px}
.scene-hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
 padding-bottom:7px;margin-bottom:12px;border-bottom:1px solid var(--hair)}
.scene-name{font-weight:600;font-size:16px}
.scene-prompt{font-style:italic;color:var(--mut);font-size:14.5px}

.matrix{display:grid;gap:9px 12px;align-items:center}
.colhead{font:500 11px/1.3 "IBM Plex Mono",monospace;letter-spacing:.06em;
 text-transform:uppercase;color:var(--mut);padding-bottom:2px;text-align:center}
.rowlab{font:500 11px/1.35 "IBM Plex Mono",monospace;letter-spacing:.05em;
 text-transform:uppercase;color:var(--mut);text-align:right;padding-right:3px}
.cell{border:1px solid var(--rule);background:var(--well);line-height:0}
.cell video{width:100%;display:block;background:var(--well)}
.cell.empty{border:1px dashed var(--rule);background:transparent;aspect-ratio:2/1;
 display:flex;align-items:center;justify-content:center;line-height:1.3}
.cell.empty span{font:400 10px/1 "IBM Plex Mono",monospace;letter-spacing:.06em;
 text-transform:uppercase;color:var(--mut);opacity:.65}
.empty{border:0;background:none}

figcaption.note{margin:16px 0 0;max-width:78ch;font-size:14px;color:var(--ink)}
figcaption.note b{font-weight:600}
footer{padding:32px 0 78px;color:var(--mut);font-size:13.5px;border-top:2px solid var(--rule);
 margin-top:38px}
footer p{max-width:78ch;margin:0 0 11px}
@media (max-width:760px){.matrix{gap:7px 6px}.rowlab{font-size:9.5px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

HEAD = ('<meta charset="utf-8">\n<title>DynaMesh Supplementary Results</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@'
        '0,8..60,400..700;1,8..60,400&display=swap">\n'
        f'<style>{CSS}</style>')


def video_cell(src):
    # src_of() may decline a clip (the preview has a size budget). An empty src would
    # render as a broken player, so an omitted cell becomes a labelled placeholder
    # that still holds its position in the matrix -- the grid must stay aligned or the
    # column headers stop meaning anything.
    if not src:
        return '<div class="cell empty"><span>in full version</span></div>'
    return (f'<div class="cell"><video src="{src}" controls loop muted playsinline '
            f'preload="metadata"></video></div>')


def render(figs, scope_html, src_of):
    """figs: list of dicts. src_of(filename) -> the src attribute to emit."""
    p = [HEAD, '<div class="wrap">', '<div class="title">',
         '<div class="venue">3DV 2027 Submission #92 &middot; Confidential review copy '
         '&middot; Do not distribute</div>',
         '<h1>DynaMesh: Dynamic 3D Texture Generation</h1>',
         '<p class="byline">Supplementary video results &middot; Anonymous 3DV submission</p>',
         f'<div class="scope">{scope_html}</div>', '</div>',
         '<nav class="toc"><h2>Contents</h2><ol>' + ''.join(
             f'<li><span class="n">Fig. {f["num"]}</span>'
             f'<a href="#{f["id"]}">{f["head"].rstrip(".")}</a></li>' for f in figs)
         + '</ol></nav>', '<main>']

    for f in figs:
        p.append(f'<section id="{f["id"]}">')
        p.append(f'<div class="fignum">Figure {f["num"]}</div>')
        p.append(f'<h2 class="head">{f["head"]}</h2>')
        p.append(f'<p class="intro">{f["intro"]}</p>')
        for sc in f['scenes']:
            cols = sc['cols']
            ncol = len(cols)
            p.append('<div class="scene"><div class="scene-hd">')
            if sc.get('name'):
                p.append(f'<span class="scene-name">{sc["name"]}</span>')
            if sc.get('prompt'):
                p.append(f'<span class="scene-prompt">“{sc["prompt"]}”</span>')
            p.append('</div>')
            p.append(f'<div class="matrix" style="grid-template-columns:118px '
                     f'repeat({ncol},minmax(0,1fr))">')
            p.append('<div></div>' + ''.join(f'<div class="colhead">{c}</div>' for c in cols))
            for lab, cells in sc['rows']:
                p.append(f'<div class="rowlab">{lab}</div>')
                for c in cells:
                    p.append(video_cell(src_of(c)) if c else '<div class="empty"></div>')
            p.append('</div></div>')
        p.append(f'<figcaption class="note"><b>Figure {f["num"]}. {f["head"]}</b> '
                 f'{f["caption"]}</figcaption>')
        p.append('</section>')

    p.append('<footer>'
             '<p><b>Reading the clips.</b> Reference videos are the raw output of the video '
             'model, that is, the input to our pipeline rather than a result. Every result clip '
             'is a composite whose left half is frozen TRELLIS.2 run per frame and whose right '
             'half is DynaMesh. Both halves are drawn into a single canvas per frame, so they '
             'share one camera and one frame index by construction, and any difference between '
             'them is attributable to the method rather than to viewpoint or timing.</p>'
             '<p>Sequences are 150 frames, except the pumpkin at 121. Scan down a column to hold '
             'the viewpoint fixed; scan across a row to hold the method fixed. No author names, '
             'institutions or file paths appear on this page or in any clip.</p>'
             '</footer></main></div>')
    return '\n'.join(p)
