"""build_daily_video_arms.py -- a daily panel video with an ARBITRARY arm list.

build_daily_video.py hardcodes the two batch-C/F tags (view_<obj>_27_<view> and
view_<obj>_27m_<view>). The batch F/G renders submitted by submit_batchfg_renders.sh
are named view_<obj>_<arm>_<view> with arm in {r19, r27, r27mcfm}, so that script
cannot see them at all. Everything else here -- the 2-panel crop, the dimmed GT
column, the label bar, the ffmpeg call -- is the same code and the same numbers, so a
batch-G daily sits beside a batch-F one without looking foreign.

    python build_daily_video_arms.py <obj> --arms r19,r27,r27mcfm [views...]

Each render is frozen|adapted. The frozen half is IDENTICAL in every arm (same frozen
model, same camera), so it is taken once from the first arm and the others are asserted
to match on frame 0 rather than assumed to.
"""
import subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageChops, ImageStat

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
LAB, BAR = 28, 30
LABELS = {'r19': 'rung19', 'r27': 'rung27', 'r27mcfm': 'rung27 + MCFM v2_D',
          'r19mcfm': 'rung19 + MCFM v2_D', 'w5': 'MCFM v2_E (W=5)',
          'stD': 'MCFM st_D', 'v3D': 'MCFM v3_D'}
DIM = []


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


def build(obj, arms, view, nfr=150, fps=25):
    seq = {a: sorted((E / f'out/view_{obj}_{a}_{view}/frames').glob('*.png')) for a in arms}
    for a, fs in seq.items():
        assert len(fs) >= nfr, f'{obj}/{a}/{view}: have {len(fs)}, need {nfr}'
    gtd = E / f'out/gt_targets_{obj}/frames'
    cols = ['ground truth', 'frozen TRELLIS.2'] + [LABELS.get(a, a) for a in arms]

    # GATE-frozen. The left panel is the frozen model, so it must agree across every
    # arm that feeds it the SAME conditioning -- if it does not, the column labelled
    # 'frozen' is really one arm's output and the comparison is a lie.
    #
    # An MCFM arm is the documented exception: render_arm.py blends the DINOv3 tokens
    # BEFORE the frozen model runs, so its left panel is frozen+MCFM, not frozen.
    # Measured here at 2.6/255 mean, 13/255 max -- small, but not zero, and averaging
    # it into the frozen column would quietly show a blended baseline. So the frozen
    # column is taken from a NON-MCFM arm and the MCFM delta is reported, not asserted
    # away. If every arm is an MCFM arm the column is frozen+MCFM and says so.
    plain = [a for a in arms if 'mcfm' not in a.lower()]
    fz_arm = plain[0] if plain else arms[0]
    if not plain:
        cols[1] = 'frozen + MCFM'
    # TOLERANCE, not equality. The renders of one view run on whatever nodes Slurm
    # hands out, and flash-attn / flex_gemm are not bit-deterministic across GPU
    # models, so two frozen renders of the SAME camera on r004 and d001 differ by
    # 0.009/255 mean. The failure this gate exists to catch -- a wrong camera or a
    # mislabelled arm -- lands at 2 to 60/255, three orders of magnitude away, so a
    # 0.25/255 bar separates them with room to spare rather than failing on noise.
    ref = panels(seq[fz_arm][0])[0]
    for a in plain:
        st = ImageStat.Stat(ImageChops.difference(ref, panels(seq[a][0])[0]))
        m = sum(st.mean) / 3
        assert m < 0.25, (f'{obj}/{view}: frozen panel of {a} differs from {fz_arm} by '
                          f'{m:.3f}/255 mean -- too large for kernel nondeterminism, '
                          f'check the camera in both render logs')
        if m:
            print(f'  frozen panel of {a} vs {fz_arm}: {m:.3f}/255 mean '
                  f'(kernel nondeterminism across nodes, under the 0.25 bar)', flush=True)
    for a in (a for a in arms if a not in plain):
        st = ImageStat.Stat(ImageChops.difference(ref, panels(seq[a][0])[0]))
        print(f'  frozen panel of {a} differs from {fz_arm} by {sum(st.mean)/3:.2f}/255 '
              f'mean (expected: MCFM blends the conditioning before the frozen model)',
              flush=True)

    tmp = E / 'out' / 'DAILY_TMP' / f'{obj}_{view}'
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(nfr):
        fz, _, s = panels(seq[fz_arm][i])
        adapted = [panels(seq[a][i])[1] for a in arms]
        if view == 'train':
            gt = Image.open(gtd / f'gt_{i+1:04d}.png').convert('RGB').resize((s, s), Image.LANCZOS)
        else:
            gt = DIM[i]
        strip = [gt, fz] + adapted
        assert len(strip) == len(cols)
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
                    '-i', str(tmp / '%04d.png'), '-vf', 'scale=1600:-2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '30',
                    '-preset', 'slow', '-movflags', '+faststart', str(out)], check=True)
    for f in tmp.glob('*.png'):
        f.unlink()
    tmp.rmdir()
    print(f'{obj:24s} {view:6s} {len(cols)} cols -> {out.name}  '
          f'{out.stat().st_size/1024:.0f} KB  ({time.time()-t0:.0f}s)', flush=True)
    return out


if __name__ == '__main__':
    argv = sys.argv[1:]
    arms = ['r19', 'r27', 'r27mcfm']
    if '--arms' in argv:
        i = argv.index('--arms')
        arms = argv[i + 1].split(',')
        argv = argv[:i] + argv[i + 2:]
    obj = argv[0]
    views = argv[1:] or ['train', 'diagA', 'diagB', 'diagC']
    print(f'{obj}: {len(arms)+2}-column daily, arms={arms}', flush=True)
    if any(v != 'train' for v in views):
        probe = sorted((E / f'out/view_{obj}_{arms[0]}_diagA/frames').glob('*.png'))[0]
        build_dim(obj, Image.open(probe).size[1] - LAB, 150)
    for v in views:
        build(obj, arms, v)
    print('ALL DONE', flush=True)
