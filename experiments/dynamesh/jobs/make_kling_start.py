"""make_kling_start.py — cut the Kling start frame out of a fixview render.

render_rung27_orbit.py writes [frozen | ours] composites with a LAB=28 label bar on
top (render_rung27_orbit.py:510-519). The Kling start frame is the RIGHT panel with
that bar removed, so this crops rather than re-renders.

The panel arithmetic is ASSERTED, not assumed: (H - LAB) * 2 must equal W. If the
composite layout ever changes, this dies instead of silently writing the wrong half
of the image into a prompt input.

usage: python jobs/make_kling_start.py [obj ...]     (default: all four figure objects)
"""
import sys
from pathlib import Path
from PIL import Image

LAB = 28
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OUT = E / 'out' / 'KLING_START'
DEFAULT = ['hand_rorschach', 'pumpkin_rot', 'chair_real_wooden_crack', 'plane_waves']

# Most objects were rendered by jobs/fixview_arm.sbatch into out/fixview_<obj>_mcfm_yaw0.
# plane_waves predates that and lives under the older view_<obj>_mcfm_train tag, which is
# the SAME training view and the SAME [frozen | ours] composite -- only the directory
# differs. Mapped here rather than re-rendering a frame that already exists.
DIRS = {'plane_waves': 'view_plane_waves_mcfm_train'}

OUT.mkdir(parents=True, exist_ok=True)
rc = 0
for o in (sys.argv[1:] or DEFAULT):
    sub = DIRS.get(o, f'fixview_{o}_mcfm_yaw0')
    frames = sorted((E / 'out' / sub / 'frames').glob('*.png'))
    if not frames:
        print(f'{o:26s} NO FRAMES — render not done'); rc = 1; continue
    src = frames[-1]
    im = Image.open(src)
    W, H = im.size
    side = H - LAB
    if side * 2 != W:
        print(f'{o:26s} GATE-layout FAILED: (H-{LAB})*2 = {side*2} != W = {W}'); rc = 1; continue
    ours = im.crop((side, LAB, 2 * side, LAB + side))     # right panel = rung27 + MCFM
    dst = OUT / f'{o}_r27_mcfmv2_D_last.png'
    ours.save(dst)
    print(f'{o:26s} {src.name}  {W}x{H} -> {ours.size[0]}x{ours.size[1]}  {dst.name}')
sys.exit(rc)
