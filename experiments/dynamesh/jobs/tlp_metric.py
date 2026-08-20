"""tlp_metric.py — tLP (temporal LPIPS), the TecoGAN temporal metric.

WHY tLP AND NOT PLAIN E_warp. Lai et al. (ECCV 2018) define the standard warping
error for BLIND TEMPORAL CONSISTENCY, where the content is static and every
frame-to-frame change is an artefact. Its long-term term E_pair(O_t, O_1)
penalises any drift from frame 1 at all. That assumption is wrong here: a dynamic
texture is SUPPOSED to change, and under Lai's long-term term the best possible
score would go to a texture that never moves.

TecoGAN (Chu et al., ACM TOG 2020, arXiv:1811.09393) solves exactly this for
generated video where content legitimately evolves. tLP measures the perceptual
distance between CONSECUTIVE frames and compares it to the REFERENCE's:

    tLP = mean_t  | LPIPS(G_{t-1}, G_t) - LPIPS(P_{t-1}, P_t) |

So a prediction is penalised for changing FASTER than the target (jitter) and for
changing SLOWER (over-smoothing / frozen texture) symmetrically. It cannot be
gamed by damping the animation, which is the objection the raw flicker number
invites.

LPIPS here is the same AlexNet the training loss uses, so the metric and the
objective are on one scale rather than two.

Reported at yaw 0, the only view with a target.
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import lpips as _lp

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
O = E / 'out'
BAR, RES = 28, 518
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OBJS = [('spot_lava', '', 150), ('teapot_lava2', '', 150),
        ('horse_metal', '_guan', 121), ('penguin_circuits', '_guan', 121),
        ('pumpkin_rot', '_guan', 121), ('whale_spots', '_guan', 121),
        ('teapot_porcelain', '_guan', 121), ('teapot_ceramic_crack', '_guan', 121)]

net = _lp.LPIPS(net='alex', verbose=False).to(DEV).eval()
for p in net.parameters():
    p.requires_grad_(False)


def t(a):
    """HWC [0,1] -> 1CHW in [-1,1], AlexNet's domain (same as the training loss)."""
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)
    return x * 2 - 1


def adapted(p):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.0
    return a[BAR:, a.shape[1] // 2:]


@torch.no_grad()
def seq_lpips(frames):
    """LPIPS between each consecutive pair."""
    out = []
    prev = t(frames[0])
    for f in frames[1:]:
        cur = t(f)
        out.append(float(net(prev, cur).mean()))
        prev = cur
    return np.array(out)


res = {}
for obj, suf, N in OBJS:
    gd = O / f'gt_targets_{obj}{suf}/frames'
    G = [np.asarray(Image.open(gd / f'gt_{i:04d}.png').convert('RGB')
                    .resize((RES, RES), Image.LANCZOS), np.float32) / 255.0
         for i in range(1, N + 1)]
    lg = seq_lpips(G)
    row = {'gt_consecutive_lpips': float(lg.mean())}
    for arm in ('r27', 'mcfm'):
        fr = sorted((O / f'fixview_{obj}_{arm}_yaw0/frames').glob('*.png'))[:N]
        if len(fr) < N:
            print(f'  skip {obj} {arm}: {len(fr)}/{N}'); continue
        lp = seq_lpips([adapted(p) for p in fr])
        row[arm] = {'tLP': float(np.abs(lp - lg).mean()),
                    'consecutive_lpips': float(lp.mean())}
    if 'r27' in row and 'mcfm' in row:
        a, b = row['r27']['tLP'], row['mcfm']['tLP']
        row['delta_pct'] = 100 * (b - a) / a
        print(f"  {obj:<22} tLP r27={a:.5f}  mcfm={b:.5f}  {row['delta_pct']:+.2f}%"
              f"   (GT consec LPIPS {row['gt_consecutive_lpips']:.5f})", flush=True)
    res[obj] = row

(O / 'tlp_metric.json').write_text(json.dumps(res, indent=1))
d = [v['delta_pct'] for v in res.values() if 'delta_pct' in v]
print(f"\n  tLP improved (lower) in {sum(1 for x in d if x < 0)}/{len(d)}   mean {np.mean(d):+.2f}%")
print('wrote', O / 'tlp_metric.json')
