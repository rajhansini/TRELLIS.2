"""gt_flicker_all42.py -- flicker against the REFERENCE VIDEO, all 42 objects.

Raw flicker rewards doing nothing: a method that barely animates scores best. On the two
objects where our texel flicker rose (napolean_waves, tie_fighter_bw) the reference clip
itself flickers MORE than either model, so frozen's lower score is under-animation, not
stability. This measures the quantity that actually matters -- how far each method's
flicker sits from the reference's.

    F(X) = mean_t mean_mask |X_t - X_{t-1}|          (temporal_metrics.py, identity warp:
                                                      the camera is locked, so nothing
                                                      but texture moves)
    error(method) = |F(method) - F(GT)|

Supervised view only: that is the one camera the reference video exists at, so it is the
only place a GT flicker can be computed at all. There is no texel-space GT -- the
reference is a video, not a textured mesh.

Panels are frozen | adapted with a 28px label bar, as everywhere else here.
"""
import glob, json, os, sys
import numpy as np
from PIL import Image

T2 = '/net/projects/ranalab/rajhansini/TRELLIS.2'
E = T2 + '/experiments/dynamesh'
os.chdir(E)
LAB, N = 28, 120
OBJS = [l.split('\t')[1] for l in open('jobs/rung37_objects.tsv').read().splitlines()[1:] if l.strip()]


def flicker(fr, m):
    return float(np.mean([np.abs(fr[t] - fr[t - 1])[m].mean() for t in range(1, len(fr))]))


def gt_frames(obj, n):
    out = []
    for i in range(1, n + 1):
        p = f'{T2}/data/{obj}/frames_from_video/frame_{i:04d}.png'
        if not os.path.exists(p):
            break
        out.append(np.asarray(Image.open(p).convert('RGB').reduce(2), dtype=np.float32) / 255.)
    return out


def panel(obj, n, side):
    out = []
    for p in sorted(glob.glob(f'{E}/out/view_{obj}_27m_train/frames/*.png'))[:n]:
        im = Image.open(p); w, h = im.size; s = h - LAB
        box = (0, LAB, s, LAB + s) if side == 'frozen' else (s, LAB, 2 * s, LAB + s)
        out.append(np.asarray(im.crop(box).convert('RGB'), dtype=np.float32) / 255.)
    return out


rows, closer, lower = [], 0, 0
for k, o in enumerate(OBJS, 1):
    g, fz, ou = gt_frames(o, N), panel(o, N, 'frozen'), panel(o, N, 'ours')
    if len(g) < 3 or len(fz) < 3 or len(ou) < 3:
        print(f'  SKIP {o}: gt={len(g)} frozen={len(fz)} ours={len(ou)}'); continue
    n = min(len(g), len(fz), len(ou))
    fg = flicker(g[:n],  np.mean(g[0], -1) < 0.93)
    ff = flicker(fz[:n], np.mean(fz[0], -1) < 0.93)
    fo = flicker(ou[:n], np.mean(ou[0], -1) < 0.93)
    ef, eo = abs(ff - fg), abs(fo - fg)
    closer += eo < ef
    lower  += fo < ff
    rows.append(dict(obj=o, gt=fg, frozen=ff, ours=fo, err_frozen=ef, err_ours=eo))
    print(f'{k:3d}/42 {o:26s} gt {fg:.5f}  frozen {ff:.5f}  ours {fo:.5f}   '
          f'|err| {ef:.5f} -> {eo:.5f} {"OURS" if eo < ef else "frozen"}', flush=True)

n = len(rows)
mef = float(np.mean([r['err_frozen'] for r in rows]))
meo = float(np.mean([r['err_ours'] for r in rows]))
print(f'\nobjects {n}')
print(f'closer to GT flicker : ours {closer}/{n}')
print(f'lower raw flicker    : ours {lower}/{n}')
print(f'mean |F - F_GT|      : frozen {mef:.5f}   ours {meo:.5f}   ({100*(1-meo/mef):.1f}% smaller)')
json.dump(dict(n=n, closer_to_gt=closer, lower_raw=lower, mean_err_frozen=mef,
               mean_err_ours=meo, per_object=rows),
          open('out/FLICKER/gt_flicker_all42.json', 'w'), indent=1)
print('wrote out/FLICKER/gt_flicker_all42.json')
