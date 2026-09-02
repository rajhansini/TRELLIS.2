"""build_batchg_pages.py -- the two batch-G dailies pages, same format as batch F.

Reuses jobs/build_daily_page.build() so the CSS tokens, the tab bar, the camera line
and selectView() are literally the same code as the pages already in the gallery.
Only the copy differs: batch G has FIVE columns (an r19 column batch F does not) and
its clips start from the grey mesh, so the continuation sentence is replaced.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_daily_page as B

GAL = 'https://claude.ai/code/artifact/42db3ff0-0cc5-4ce7-92e1-0a03fec3b293'
COLS = ('<b>Five columns</b> &mdash; ground truth, frozen TRELLIS.2, rung19 '
        '(cross-attention LoRA), rung27 (+ self-attention LoRA), and rung27 + MCFM '
        'temporal-only (3-frame window, <span class="mono">v2_D</span>). Batch G ran '
        'all three arms, so the ladder is readable left to right within one clip.')
ORIGIN = ('<b>This clip starts from the grey mesh.</b> Kling was handed the untextured '
          'unicorn render and the texture ARRIVES over the 150 frames &mdash; frame 1 is '
          'bare geometry, not a textured still being kept in motion.')
ALIGN = ('<b>Alignment</b> &mdash; solved by <span class="mono">solve_orientation.py</span> '
         '(24 cube rotations then coordinate descent), silhouette IoU 0.948 and 0.947 '
         'against our render, 1.0% uncovered. No scale or shift correction was applied '
         'to either clip.')

OBJS = [
    ('unicorn_rainbow', 'Unicorn &mdash; Rainbow', 'unicorn bust',
     'iridescent blue-to-green sheen sweeping the body'),
    ('unicorn_effect1', 'Unicorn &mdash; Cracked Glaze', 'unicorn bust',
     'dark glaze veined with glowing amber cracks'),
    ('car_effect1', 'Car &mdash; Charred Grain', 'nascar',
     'burnt wood grain running front to back over the body'),
    ('car_effect2', 'Car &mdash; Oil Slick', 'nascar',
     'wet black lacquer flecked with pale chips'),
    ('robot_rust_2', 'Robot &mdash; Rust', 'robot',
     'oxide creeping over the limbs and torso'),
    ('pig_rainbow', 'Pig &mdash; Rainbow', 'pig',
     'saturated colour sweeping the body'),
    ('pig_red', 'Pig &mdash; Red', 'pig',
     'red spreading over the hide'),
]

# The cars sit at align IoU 0.898 -- 0.002 under the bar the gargoyle failure set --
# so their pages say so rather than presenting them as clean. robot_rust_2 at 0.939 is
# the regeneration that fixed the clip batch G originally rejected at 0.612.
PIG_ALIGN = ('<b>Alignment &mdash; this clip was rejected once, wrongly.</b> The first solve '
             'searched rotation ALONE, with scale fixed, and reported IoU 0.711; the clip was '
             'written up as "silhouette redrawn". It was not. Our render was 26% too small, so '
             'the rotation search was scoring poses at the wrong size. Fitting rotation and a '
             '2D similarity together lands it at <span class="mono">{iou}</span> &mdash; better '
             'than any other object in this batch. Frames are resampled by s&#8776;0.796; the '
             'untouched upload is kept in <span class="mono">frames_from_video_raw</span>.')

ALIGN_BY_OBJ = {
    'pig_rainbow': PIG_ALIGN.format(iou='0.974'),
    'pig_red': PIG_ALIGN.format(iou='0.962'),
    'car_effect1': ('<b>Alignment, read this before judging the result.</b> This clip solved '
                    'at silhouette IoU <span class="mono">0.898</span> with 4.6% of the mesh '
                    'outside the video, AFTER a similarity correction (scale 0.900, dx +9, '
                    'dy &minus;11) was applied to all 150 frames. That is two thousandths '
                    'under the 0.90 bar the gargoyle failure set, so treat it as provisional.'),
    'car_effect2': ('<b>Alignment, read this before judging the result.</b> Same camera error '
                    'and same correction as the other car clip: IoU <span class="mono">0.898</span>, '
                    '4.4% uncovered, scale 0.900. Two thousandths under the 0.90 bar.'),
    'robot_rust_2': ('<b>Alignment</b> &mdash; IoU <span class="mono">0.939</span>, 2.0% '
                     'uncovered. This is the REGENERATED clip: the first robot upload came '
                     'back with the limbs redrawn and failed at 0.612, which no scale or '
                     'shift correction can fix. This one clears the bar outright.'),
}

if __name__ == '__main__':
    only = sys.argv[1:]
    for obj, title, mesh, effect in OBJS:
        if only and obj not in only:
            continue
        B.build(obj, title, mesh, effect, GAL,
                cols_line=COLS, origin_line=ORIGIN,
                legend_extra=f'<div>{ALIGN_BY_OBJ.get(obj, ALIGN)}</div>',
                eyebrow='Batch G Dailies',
                out=f'/net/projects/ranalab/rajhansini/TRELLIS.2/dailies_{obj}.html')
