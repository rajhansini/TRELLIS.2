"""Contact sheet: every mesh rendered under all six up-axis assignments.

Maximising exposed area ALONE picks nonsense poses -- it happily lays a dog on its
back, because a belly-up view exposes more surface than a standing one. So the
up-axis is chosen by eye from these sheets, and only the YAW is then optimised for
exposure. Meshes here come from four unrelated collections with no shared frame,
so there is no convention to rely on.
"""
import sys, pathlib, numpy as np
from PIL import Image, ImageDraw
import multiprocessing as mp

HERE = pathlib.Path(__file__).resolve().parent
g = {'__file__': str(HERE / 'best_view.py'), '__name__': 'lib'}
exec(open(HERE / 'best_view.py').read().split('YAWS = range')[0], g)
load_obj, normalize, sample, render, UPS = (g['load_obj'], g['normalize'],
                                            g['sample'], g['render'], g['UPS'])
SH = HERE / 'orient'; SH.mkdir(exist_ok=True)
TH = 240

def one(path):
    V, F = load_obj(path); V = normalize(V)
    tiles = []
    for uname, U in UPS.items():
        P, N = sample(V @ U.T, F, 250_000, seed=1)
        im = Image.fromarray(render(P, N, 30, 8, res=TH * 2)).convert('RGB')
        tiles.append((uname, im.resize((TH, TH), Image.LANCZOS)))
    row = Image.new('RGB', (TH * 6, TH + 26), 'white')
    d = ImageDraw.Draw(row)
    for i, (uname, im) in enumerate(tiles):
        row.paste(im, (i * TH, 26)); d.text((i * TH + 6, 6), uname, fill='black')
    d.text((4, TH + 8), path.stem, fill='black')
    row.save(SH / f'{path.stem}.png')
    return path.stem

if __name__ == '__main__':
    todo = sorted((HERE / 'OBJ').glob('*.obj'))
    with mp.Pool(8) as pool:
        for n in pool.imap_unordered(one, todo):
            print('sheet', n, flush=True)
    # stack rows into a few readable sheets
    rows = sorted(SH.glob('*.png'))
    for k in range(0, len(rows), 6):
        ims = [Image.open(p) for p in rows[k:k + 6]]
        W = max(i.width for i in ims); H = sum(i.height for i in ims)
        s = Image.new('RGB', (W, H), 'white'); y = 0
        for i in ims: s.paste(i, (0, y)); y += i.height
        s.save(HERE / f'orient_sheet_{k//6}.png')
    print('sheets:', len(rows))
