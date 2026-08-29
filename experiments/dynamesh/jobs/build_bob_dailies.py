"""build_bob_dailies.py -- the three bob (duck) continuation dailies pages.

Three assets share ONE mesh -- Keenan Crane's bob -- and differ only in the driving
Kling clip and, for bob_spots_front, the camera the texture was fitted at. So they are
three pages in the same gallery rather than one page with three objects: the view tabs
are per-object, and stacking three objects' worth of base64 video on one page walks
straight into the 16 MB artifact cap.

THREE COLUMNS, NOT FOUR. These were trained on the MCFM arm only, so the plain-rung27
column does not exist and the legend says three. See build_daily_video.py --no-r27.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_daily_page import build                      # one source of truth for the markup

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
GALLERY = 'https://claude.ai/code/artifact/42db3ff0-0cc5-4ce7-92e1-0a03fec3b293'
OUT = E / 'out' / 'BOBDAILIES'

COLS3 = ('<b>Three columns</b> &mdash; ground truth, frozen TRELLIS.2, rung27 + MCFM '
         'temporal-only (3-frame window, <span class="mono">v2_D</span>). There is no '
         'plain-rung27 column: these assets were trained on the MCFM arm only.')

OBJS = [
    ('bob_spots', 'Bob — Spots',
     'blue and white polka dots gliding over the yellow duck'),
    ('bob_spots_slow', 'Bob — Spots, Slow',
     'the same drift at a lower rate — the v2 prompt, before the speed floor'),
    ('bob_spots_front', 'Bob — Spots, Front View',
     'yaw 250 conditioning: both eyes and the beak inside the supervised view'),
]


def psnr(obj):
    """(frozen, adapted) from the object's own training log, or None."""
    best = None
    for p in sorted((E / 'out/FIGRUNS').glob(f'27_v2_D_{obj}_*.log')):
        t = p.read_text(errors='replace')
        if f'OBJ={obj} ' not in t:
            continue
        f = re.search(r'\[FINAL\] frozen\s+PSNR ([\d.]+)\s+SSIM ([\d.]+)', t)
        a = re.search(r'\[FINAL\] rung27\s+PSNR ([\d.]+)\s+SSIM ([\d.]+)', t)
        if f and a:
            best = (float(f.group(1)), float(f.group(2)),
                    float(a.group(1)), float(a.group(2)))
    return best


def table(obj):
    v = psnr(obj)
    if not v:
        return ('<div class="note"><b>Training numbers not in yet.</b> The page is built '
                'from the rendered panels; the table appears once the run logs its '
                '<span class="mono">[FINAL]</span> line.</div>')
    fp, fs, ap_, as_ = v
    return (f'<h2>At the training view</h2>'
            f'<table class="num"><thead><tr><th>arm</th><th>PSNR</th><th>SSIM</th></tr></thead>'
            f'<tbody>'
            f'<tr><td>frozen TRELLIS.2</td><td>{fp:.3f}</td><td>{fs:.4f}</td></tr>'
            f'<tr class="hi"><td>rung27 + MCFM v2_D</td><td>{ap_:.3f}</td><td>{as_:.4f}</td></tr>'
            f'<tr><td>delta</td><td>{ap_-fp:+.3f} dB</td><td>{as_-fs:+.4f}</td></tr>'
            f'</tbody></table>')


def nav(cur):
    return ('<div class="objnav">' + ''.join(
        f'<span class="pill{" current" if o == cur else ""}">{t}</span>'
        for o, t, _ in OBJS) + '</div>')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or [o for o, *_ in OBJS]
    for obj, title, effect in OBJS:
        if obj not in want:
            continue
        missing = [v for v in ('train', 'diagA', 'diagB', 'diagC')
                   if not (E / 'out/DAILY' / f'{obj}_{v}.mp4').exists()]
        if missing:
            print(f'SKIP {obj}: missing videos {missing}')
            continue
        p = build(obj, title, 'bob (Keenan Crane)', effect, GALLERY,
                  extra_rows=table(obj) + nav(title),
                  cols_line=COLS3,
                  eyebrow='Bob Continuation Dailies',
                  out=OUT / f'{obj}_daily.html')
        print(f'  {obj}: {p}')
