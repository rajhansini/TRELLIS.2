"""vase_grid.py -- one labelled contact sheet of the whole vase sweep.

Renders are 1024^2 and mostly white, so every tile is cropped to the UNION
silhouette bbox across all views first (same crop for all, so relative size
stays honest) -- that is what makes the vase big enough to judge.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = Path("/net/projects/ranalab/rajhansini/TRELLIS.2/data/dynamesh_meshes/vase_sweep")
TW, TH, PAD = 150, 300, 6
BEST_YAW, BEST_PITCH = 240, 25

M = {}
for ln in open(D/"metrics.tsv").read().splitlines()[1:]:
    p = ln.split("\t"); M[p[0]] = (float(p[1]), float(p[2]), float(p[4]), float(p[5]))

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists(): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F1, F2, F3 = font(17), font(14), font(26)

yaws   = [f"yaw_{y:03d}"   for y in range(0, 360, 10)]
pitches= [f"pitch_{p:02d}" for p in (0, 5, 10, 15, 20, 25, 30, 40)]
imgs   = {t: Image.open(D/f"{t}.png").convert("RGB") for t in yaws+pitches}

# union bbox of everything non-white -> one shared crop
b = None
for im in imgs.values():
    a = np.array(im).min(2)
    ys, xs = np.where(a < 245)
    q = [xs.min(), ys.min(), xs.max(), ys.max()]
    b = q if b is None else [min(b[0],q[0]), min(b[1],q[1]), max(b[2],q[2]), max(b[3],q[3])]
b = [b[0]-12, b[1]-12, b[2]+12, b[3]+12]

def tile(tag, best):
    yaw, pitch, vis, hand = M[tag]
    t = Image.new("RGB", (TW, TH+34), "white")
    t.paste(imgs[tag].crop(b).resize((TW-2*PAD, TH-2*PAD)), (PAD, 28+PAD))
    d = ImageDraw.Draw(t)
    lab = f"{yaw:.0f}°" if tag.startswith("yaw") else f"pitch {pitch:.0f}°"
    if tag.startswith("yaw"):
        d.text((PAD, 4), lab, fill=(200,0,0) if best else (0,0,0), font=F1)
        d.text((PAD+62, 7), f"{vis:.1f} | {hand:.0f}", fill=(90,90,90), font=F2)
    else:
        d.text((PAD, 1), lab, fill=(200,0,0) if best else (0,0,0), font=F1)
        d.text((PAD, 16), f"{vis:.1f} | {hand:.0f}", fill=(90,90,90), font=F2)
    if best:
        d.rectangle([0,0,TW-1,TH+33], outline=(200,0,0), width=3)
    return t

def grid(tags, ncol, bestfn):
    ts = [tile(t, bestfn(t)) for t in tags]
    nrow = (len(ts)+ncol-1)//ncol
    g = Image.new("RGB", (ncol*TW, nrow*(TH+34)), "white")
    for i, t in enumerate(ts):
        g.paste(t, ((i % ncol)*TW, (i//ncol)*(TH+34)))
    return g

gy = grid(yaws, 9, lambda t: M[t][0] == BEST_YAW)
gp = grid(pitches, 8, lambda t: M[t][1] == BEST_PITCH)

HDR = 46
W = max(gy.width, gp.width)
out = Image.new("RGB", (W, HDR + gy.height + HDR + gp.height + 10), "white")
d = ImageDraw.Draw(out)
d.text((8, 10), "vase.obj (one handle) — yaw sweep @ pitch 10°"
                "     label: yaw°  |  visible-area%  |  handle-visible%",
       fill=(0,0,0), font=F1)
out.paste(gy, (0, HDR))
y2 = HDR + gy.height + 8
d.text((8, y2+10), f"pitch sweep @ yaw {BEST_YAW}°"
                   f"     BEST = yaw {BEST_YAW}° / pitch {BEST_PITCH}°", fill=(0,0,0), font=F1)
out.paste(gp, (0, y2+HDR))
out.save(D/"GRID.png")
print("wrote", D/"GRID.png", out.size)
