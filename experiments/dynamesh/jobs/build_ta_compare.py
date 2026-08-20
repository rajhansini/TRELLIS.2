"""build_ta_compare.py — three-mode conditioning comparison, both repos.

One row per repo, three modes across. Metric is Temporal Flickering (VBench,
Huang et al. CVPR 2024) with drift as the guard, computed on the rendered frames
because TRELLIS 1 GENERATES its geometry from a reference frame -- there is no
fixed mesh and therefore no shared voxel field or GT silhouette to measure in.

Absolute numbers do NOT transfer between repos: different backbone (TRELLIS-image-large
vs TRELLIS.2-4B), different DINO (v2/518/1374 tokens vs v3/512/1029), different
geometry handling. Only the RANKING of the three modes within a repo is comparable,
and agreement of that ranking across two pipelines is the actual result.
"""
import json, subprocess
import numpy as np
from pathlib import Path
from PIL import Image

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T1 = Path('/net/projects/ranalab/rajhansini/TRELLIS/experiments/enhancement')
OUT = E / 'out/TA_COMPARE'; OUT.mkdir(exist_ok=True)
OBJS = [('spot_lava', 150), ('hand_rorschach', 150), ('skull_lava', 150), ('pumpkin_rot', 121)]
MODES = [('temporal_only', 'v2', 'temporal only'),
         ('spatial_then_temporal', 'v2b', 'spatial then temporal'),
         ('joint_spatiotemporal', 'v3', 'joint spatiotemporal')]
BAR = 28


def frames2(o, m):
    return sorted((E / f'out/ta2_{o}_{m}_w3/frames').glob('*.png'))


def frames1(o, v):
    return sorted((T1 / f'results_mcfm_ta1_{o}_{v}_phase_d_{v}').rglob('*.png'))


def load(p, crop_bar):
    a = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.
    return a[BAR:] if crop_bar else a


def metrics(paths, n, crop_bar):
    """Temporal Flickering + drift, on the object only (non-white pixels)."""
    P = np.stack([load(p, crop_bar) for p in paths[:n]])
    m = (P.min(-1) < 0.96).any(0)                 # union silhouette over the clip
    if m.sum() < 100:
        m = np.ones(P.shape[1:3], bool)
    f = float(np.mean([np.abs(P[t] - P[t-1])[m].mean() for t in range(1, n)]))
    d = float(np.abs(P[-1] - P[0])[m].mean())
    return f, d, int(m.sum())


res = {}
for o, N in OBJS:
    res[o] = {}
    for mode, v, _ in MODES:
        f2 = frames2(o, mode); f1 = frames1(o, v)
        r = {}
        if len(f2) >= N:
            fl, dr, px = metrics(f2, N, True);  r['T2'] = dict(flicker=fl, drift=dr, px=px)
        if len(f1) >= N:
            fl, dr, px = metrics(f1, N, False); r['T1'] = dict(flicker=fl, drift=dr, px=px)
        res[o][mode] = r
        print(f"  {o:<16}{mode:<24} T1 F={r.get('T1',{}).get('flicker',float('nan')):.5f}"
              f"  T2 F={r.get('T2',{}).get('flicker',float('nan')):.5f}", flush=True)

(E / 'out/ta_compare.json').write_text(json.dumps(res, indent=1))

# ── videos: per object, rows = repo, cols = mode ────────────────────────────
from PIL import ImageDraw
for o, N in OBJS:
    wd = OUT / f'_w_{o}'; subprocess.run(['rm', '-rf', str(wd)]); wd.mkdir(parents=True)
    ok = all(len(frames2(o, m)) >= N and len(frames1(o, v)) >= N for m, v, _ in MODES)
    if not ok:
        print(f'  {o}: incomplete, video skipped'); continue
    F2 = {m: frames2(o, m) for m, v, _ in MODES}
    F1 = {v: frames1(o, v) for m, v, _ in MODES}
    for i in range(N):
        tiles = []
        for repo, lab in (('T1', 'TRELLIS 1'), ('T2', 'TRELLIS.2')):
            row = []
            for mode, v, nice in MODES:
                p = F1[v][i] if repo == 'T1' else F2[mode][i]
                im = Image.open(p).convert('RGB')
                if repo == 'T2':
                    im = im.crop((0, BAR, im.width, im.height))
                im = im.resize((518, 518), Image.LANCZOS)
                t = Image.new('RGB', (518, 518 + 26), (22, 23, 28))
                t.paste(im, (0, 26))
                ImageDraw.Draw(t).text((8, 7), f'{lab}   {nice}', fill=(255, 255, 255))
                row.append(t)
            tiles.append(row)
        W, H = 518 * 3, (518 + 26) * 2
        g = Image.new('RGB', (W, H), (22, 23, 28))
        for r, row in enumerate(tiles):
            for c, t in enumerate(row):
                g.paste(t, (c * 518, r * (518 + 26)))
        g.save(wd / f'{i+1:04d}.png')
    mp4 = OUT / f'ta_{o}.mp4'
    subprocess.run(f'ffmpeg -nostdin -v error -y -framerate 20 -i {wd}/%04d.png '
                   f'-vf "scale=trunc(iw/3)*2:trunc(ih/3)*2" -c:v libx264 -crf 30 -preset slow '
                   f'-pix_fmt yuv420p -movflags +faststart {mp4}', shell=True)
    subprocess.run(['rm', '-rf', str(wd)])
    print(f'  {o}: {mp4.stat().st_size//1024} KB')
print('done')
