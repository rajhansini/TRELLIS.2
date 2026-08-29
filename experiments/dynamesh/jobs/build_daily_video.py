"""build_daily_video.py -- the four-column daily: GT | frozen | rung27 | rung27+MCFM.

Built from the two frozen|adapted panels render_arm.sbatch writes:
    out/view_<obj>_27_<view>/frames    left=frozen  right=rung27
    out/view_<obj>_27m_<view>/frames   left=frozen  right=rung27+MCFM
The frozen half is identical in both, so it is taken once from the 27m panel.

GT column: at the training view it is the 2D-copy target the loss actually saw. At the
three unseen diagonals there is NO ground truth -- the camera never shot that angle -- so
the driving-video frame is shown dimmed, exactly as the batch-C dailies do.

WHY THE GT COLUMN IS CACHED. Measured on this filesystem, opening one 1440^2 driving
frame costs 1.60s and a LANCZOS resize to 518 another 1.51s. The dimmed column is
IDENTICAL at diagA/diagB/diagC, so rebuilding it per view spent 3.1s x 150 x 3 = 23
minutes redoing the same work. It is built once and reused. reduce(2) is an integer box
filter, so 1440->720 is nearly free and only the small 720->518 step pays for quality.

Panel arithmetic is asserted, never assumed.
"""
import subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
LAB, BAR = 28, 30
COLS = ['ground truth', 'frozen TRELLIS.2', 'rung27', 'rung27 + MCFM v2_D']

# THREE-COLUMN MODE (--no-r27). Some objects were trained on the MCFM arm ONLY, so
# view_<obj>_27_<view> never exists. Dropping the column is an EXPLICIT flag rather
# than an is-the-directory-there test: a silently-3-wide daily looks just as finished
# as a 4-wide one, and the reader has no way to tell a deliberate omission from a
# render that failed. Asked for, or it fails on the missing directory as before.
NO_R27 = False
DIM = []                      # the shared dimmed-GT column


def panels(p):
    im = Image.open(p)
    W, H = im.size
    s = H - LAB
    assert s * 2 == W, f'{p.name}: layout {W}x{H} is not 2 panels + {LAB}px bar'
    return im.crop((0, LAB, s, LAB + s)), im.crop((s, LAB, 2 * s, LAB + s)), s


def build_dim(obj, s, nfr):
    t0 = time.time()
    white = Image.new('RGB', (s, s), (255, 255, 255))
    for i in range(nfr):
        f = T2 / f'data/{obj}/frames_from_video/frame_{i+1:04d}.png'
        im = Image.open(f).convert('RGB').reduce(2).resize((s, s), Image.LANCZOS)
        DIM.append(Image.blend(im, white, 0.55))
    print(f'dimmed GT column: {nfr} frames in {time.time()-t0:.0f}s '
          f'(built once, shared by all three diagonals)', flush=True)


def build(obj, view, nfr=150, fps=25):
    a27 = [] if NO_R27 else sorted((E / f'out/view_{obj}_27_{view}/frames').glob('*.png'))
    a27m = sorted((E / f'out/view_{obj}_27m_{view}/frames').glob('*.png'))
    gtd = E / f'out/gt_targets_{obj}/frames'
    cols = [c for c in COLS if not (NO_R27 and c == 'rung27')]
    assert len(a27m) >= nfr and (NO_R27 or len(a27) >= nfr), \
        f'{obj}/{view}: have {len(a27)}/{len(a27m)} frames, need {nfr}'
    tmp = E / 'out' / 'DAILY_TMP' / f'{obj}_{view}'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(nfr):
        fz, mc, s = panels(a27m[i])
        r27 = None if NO_R27 else panels(a27[i])[1]
        if view == 'train':
            gt = Image.open(gtd / f'gt_{i+1:04d}.png').convert('RGB').resize((s, s), Image.LANCZOS)
        else:
            gt = DIM[i]
        strip = [gt, fz, mc] if NO_R27 else [gt, fz, r27, mc]
        canv = Image.new('RGB', (s * len(strip), s + BAR), (22, 22, 26))
        d = ImageDraw.Draw(canv)
        for j, (im, lb) in enumerate(zip(strip, cols)):
            canv.paste(im, (j * s, BAR))
            d.rectangle([j * s, 0, (j + 1) * s - 1, BAR - 1], fill=(38, 38, 48))
            if view != 'train' and j == 0:
                lb += '   (no GT at this view)'
            d.text((j * s + 8, 9), lb, fill=(232, 232, 240))
        canv.save(tmp / f'{i:04d}.png')
    out = E / 'out' / 'DAILY' / f'{obj}_{view}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-framerate', str(fps),
                    '-i', str(tmp / '%04d.png'), '-vf', 'scale=1280:-2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '30',
                    '-preset', 'slow', '-movflags', '+faststart', str(out)], check=True)
    for f in tmp.glob('*.png'):
        f.unlink()
    tmp.rmdir()
    print(f'{obj:42s} {view:6s} -> {out.name}  {out.stat().st_size/1024:.0f} KB  '
          f'({time.time()-t0:.0f}s)', flush=True)
    return out


if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--no-r27']
    NO_R27 = '--no-r27' in sys.argv
    obj = argv[0]
    views = argv[1:] or ['train', 'diagA', 'diagB', 'diagC']
    print(f'{obj}: {"THREE" if NO_R27 else "four"}-column daily', flush=True)
    if any(v != 'train' for v in views):
        probe = sorted((E / f'out/view_{obj}_27m_diagA/frames').glob('*.png'))[0]
        build_dim(obj, Image.open(probe).size[1] - LAB, 150)
    for v in views:
        build(obj, v)
    print('ALL DONE', flush=True)
