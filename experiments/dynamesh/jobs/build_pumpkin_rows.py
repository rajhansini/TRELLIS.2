"""build_pumpkin_rows.py — GT | frozen | ours, one row per view, for the pumpkin transfer.

render_rung30_transfer.py emits TWO panels (frozen | ours). The left panel is
composited on here, which is why it keeps having to be a separate step.

WHAT THE LEFT PANEL IS. For a transfer there is no ground truth: the pumpkin was
never trained on, so no target for it exists. The left panel is the SOURCE texture
-- spot_lava's video, the thing being carried onto a new shape -- and it is
labelled as such rather than as GT. Calling it ground truth would imply a
reconstruction metric that cannot exist here.

Two outputs:
  per-view  <tag>_row.mp4        one row, 3 panels, 150 frames
  stacked   pumpkin_4views.mp4   the four fixed cameras as four rows
"""
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
O, T2 = E / 'out', Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
SRCF = sorted((O / 'gt_targets_spot_lava/frames').glob('gt_*.png'))
OUTD = O / 'R30_PUMPKIN'; OUTD.mkdir(exist_ok=True)
BAR, N = 30, 150
VIEWS = [('r30_pumpkin_yaw000', 'yaw 0deg  (source training view)'),
         ('r30_pumpkin_yaw090', 'yaw 90deg  UNSEEN'),
         ('r30_pumpkin_yaw180', 'yaw 180deg  UNSEEN'),
         ('r30_pumpkin_yaw270', 'yaw 270deg  UNSEEN'),
         ('r30_pumpkin_360',    '360deg orbit'),
         ('r30_pumpkin_720',    '720deg orbit')]

def row(tag, label, i):
    """one frame: SOURCE | frozen | ours, with the view named on it."""
    src = sorted((O / tag / 'frames').glob('*.png'))
    im = Image.open(src[i]).convert('RGB')          # already frozen|ours
    W = im.width // 2
    g = Image.open(SRCF[i]).convert('RGB').resize((W, im.height), Image.LANCZOS)
    out = Image.new('RGB', (W + im.width, im.height + BAR), (22, 23, 28))
    out.paste(g, (0, BAR)); out.paste(im, (W, BAR))
    d = ImageDraw.Draw(out)
    d.text((10, 9), f'{label}          SOURCE TEXTURE (spot_lava)  |  '
                    f'FROZEN TRELLIS.2  |  ADAPTED (zero-shot)',
           fill=(255, 210, 120) if 'yaw 0deg' in label else (255, 255, 255))
    return out

done = []
for tag, label in VIEWS:
    src = sorted((O / tag / 'frames').glob('*.png'))
    if len(src) < N:
        print(f'  {tag}: {len(src)}/{N} frames — SKIPPED'); continue
    wd = O / '_pk' / tag; subprocess.run(['rm', '-rf', str(wd)]); wd.mkdir(parents=True)
    for i in range(N):
        row(tag, label, i).save(wd / f'{i+1:04d}.png')
    mp4 = OUTD / f'{tag}_row.mp4'
    subprocess.run(f'ffmpeg -nostdin -v error -y -framerate 20 -i {wd}/%04d.png '
                   f'-c:v libx264 -crf 22 -preset slow -pix_fmt yuv420p '
                   f'-movflags +faststart {mp4}', shell=True)
    print(f'  {tag}: {mp4.stat().st_size//1024} KB')
    done.append(tag)

fixed = [t for t, _ in VIEWS[:4] if t in done]
if len(fixed) == 4:
    wd = O / '_pk4'; subprocess.run(['rm', '-rf', str(wd)]); wd.mkdir(parents=True)
    for i in range(N):
        rows = [row(t, l, i) for t, l in VIEWS[:4]]
        w, h = rows[0].size
        s = Image.new('RGB', (w, h * 4), (22, 23, 28))
        for k, r in enumerate(rows):
            s.paste(r, (0, k * h))
        s.save(wd / f'{i+1:04d}.png')
    mp4 = OUTD / 'pumpkin_4views.mp4'
    subprocess.run(f'ffmpeg -nostdin -v error -y -framerate 20 -i {wd}/%04d.png '
                   f'-c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p '
                   f'-movflags +faststart {mp4}', shell=True)
    print(f'  STACKED 4 views: {mp4.stat().st_size//1024} KB')
subprocess.run(['rm', '-rf', str(O / '_pk'), str(O / '_pk4')])
print('rows built:', len(done))
