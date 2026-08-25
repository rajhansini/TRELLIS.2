"""compose_vase_clips.py -- build the limitation-artifact clips for vase_floral.

panel_view.sbatch writes each frame as ONE 1036x546 image: a 28px label strip on
top, then two 518x518 panels -- LEFT = TRELLIS.2 frozen, RIGHT = that run's trained
model. So the frozen arm is already rendered inside both runs' output and needs no
job of its own; the two runs differ only in their right-hand panel.

Per yaw the strips are recombined into one row:
    yaw 0            GT | frozen | rung27+MCFM   (GT exists: it IS the training view)
    yaw 90/180/270        frozen | rung27+MCFM   (no GT -- the Kling video is a single
                                                  fixed camera, so no ground truth
                                                  exists at an unseen yaw; showing one
                                                  there would be a fabrication)
Encoded with the SYSTEM ffmpeg: the trellis2 env's ffmpeg has no libx264 and exits 8.
"""
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OBJ, NFR, P = 'vase_floral', 150, 518
STRIP, LBL = 28, 30
OUT = E/'out'/'VASE_CLIPS'; OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = '/usr/bin/ffmpeg'


def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'):
        if Path(p).exists(): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F = font(15)


def half(img, right):
    """Strip the baked-in label row, return one 518x518 panel."""
    x = P if right else 0
    return img.crop((x, STRIP, x+P, STRIP+P))


def build(yaw):
    md = E/f'out/panel_{OBJ}_mcfm_y{yaw}/frames'
    gd = E/'out/gt_targets_vase_floral/frames'
    for d in (md,):
        n = len(list(d.glob('*.png')))
        if n < NFR: return f'SKIP yaw{yaw}: {d.name} has {n}/{NFR}'
    use_gt = (yaw == 0)
    cols = (['ground truth (Kling)'] if use_gt else []) + \
           ['TRELLIS.2 frozen', 'rung27 + MCFM v2_D']
    W = P*len(cols)
    fdir = OUT/f'{OBJ}_yaw{yaw}_frames'; fdir.mkdir(exist_ok=True)
    for i in range(1, NFR+1):
        m = Image.open(md/f'{i:04d}.png').convert('RGB')
        tiles = []
        if use_gt:
            tiles.append(Image.open(gd/f'gt_{i:04d}.png').convert('RGB').resize((P, P)))
        tiles += [half(m, False), half(m, True)]
        canvas = Image.new('RGB', (W, P+LBL), 'white')
        d = ImageDraw.Draw(canvas)
        for k, (t, name) in enumerate(zip(tiles, cols)):
            canvas.paste(t, (k*P, LBL))
            d.text((k*P+10, 8), name, fill=(20, 20, 20), font=F)
            if k: d.line([(k*P, 0), (k*P, P+LBL)], fill=(205, 205, 205), width=1)
        canvas.save(fdir/f'{i:04d}.png')
    mp4 = OUT/f'{OBJ}_yaw{yaw}_{"GT_" if use_gt else ""}frozen_r27_mcfm.mp4'
    cmd = [FFMPEG, '-nostdin', '-v', 'error', '-y', '-framerate', '20',
           '-i', str(fdir/'%04d.png'), '-c:v', 'libx264', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)]
    subprocess.run(cmd, check=True)
    return f'OK   yaw{yaw:3d} -> {mp4.name}  ({mp4.stat().st_size/1024:.0f} KB, {len(cols)} panels)'


for y in (0, 90, 180, 270):
    print(build(y), flush=True)
