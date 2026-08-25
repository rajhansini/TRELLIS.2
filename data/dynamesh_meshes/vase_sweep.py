"""vase_sweep.py -- pick the yaw/pitch that exposes the most of the one-handle vase.

Camera is NOT reimplemented: load_norm/render are imported from guanc's
render_final.py, the exact script that produced every Kling input PNG
(res 1024, dist 2.6, fov 30, y-up, fixed cam on +Z).  raster_ids() below
duplicates that rasteriser line-for-line and additionally writes a FACE-ID
buffer, so "how much of the vase is exposed" is measured as real visible
surface area, not silhouette pixels.  _selftest() asserts the duplicated
raster is byte-identical to render_final.render() before any sweeping.

Exposure metrics per view:
  sil%      silhouette pixels / image  -- how big it sits in frame
  vis%      visible face AREA / total face area  -- the real exposure number
  hand%     visible handle area / total handle area  -- the discriminator,
            since a vase body is a surface of revolution and barely changes
            with yaw; only the handle does.
"""
import importlib.util, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

MESH = "/net/projects/ranalab/itailang/multi_iSeg/meshes/vase.obj"   # genus 1, one handle
RF   = "/net/projects/ranalab/guanc/tmp/kling/render_final.py"
OUT  = Path("/net/projects/ranalab/rajhansini/TRELLIS.2/data/dynamesh_meshes/vase_sweep")
RES, DIST, FOV = 1024, 2.6, 30.0

spec = importlib.util.spec_from_file_location("render_final", RF)
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)


def cam(yaw_deg, pitch_deg):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    eye = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])*DIST
    fwd = -eye/np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def raster_ids(m, yaw_deg, pitch_deg, res=RES):
    """render_final.render() + a face-id buffer. Returns (rgb_uint8, fid_int32)."""
    eye, right, up, fwd = cam(yaw_deg, pitch_deg)
    L1 = (-0.4*right+0.5*up-0.8*fwd); L1 /= np.linalg.norm(L1)
    L2 = ( 0.7*right+0.2*up-0.5*fwd); L2 /= np.linalg.norm(L2)
    V, F, VN = np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.vertex_normals)
    Vc = (V-eye) @ np.stack([right, up, fwd], 1)
    f = 1.0/np.tan(np.radians(FOV)/2)
    z = Vc[:, 2].clip(1e-6)
    P = np.stack([(Vc[:, 0]*f/z*0.5+0.5)*(res-1), (-Vc[:, 1]*f/z*0.5+0.5)*(res-1)], 1)
    zbuf = np.full((res, res), np.inf); img = np.ones((res, res, 3))
    fid = np.full((res, res), -1, np.int32)
    order = np.argsort(Vc[F].mean(1)[:, 2])[::-1]
    for fi in order:
        i0, i1, i2 = F[fi]
        p0, p1, p2 = P[i0], P[i1], P[i2]
        xmin = int(max(0, np.floor(min(p0[0], p1[0], p2[0])))); xmax = int(min(res-1, np.ceil(max(p0[0], p1[0], p2[0]))))
        ymin = int(max(0, np.floor(min(p0[1], p1[1], p2[1])))); ymax = int(min(res-1, np.ceil(max(p0[1], p1[1], p2[1]))))
        if xmax < xmin or ymax < ymin: continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
        d = (p1[0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[1]-p0[1])
        if abs(d) < 1e-12: continue
        w1 = ((xs-p0[0])*(p2[1]-p0[1])-(ys-p0[1])*(p2[0]-p0[0]))/d
        w2 = ((p1[0]-p0[0])*(ys-p0[1])-(p1[1]-p0[1])*(xs-p0[0]))/d
        w0 = 1-w1-w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any(): continue
        zi = w0*Vc[i0, 2]+w1*Vc[i1, 2]+w2*Vc[i2, 2]
        mask = inside & (zi < zbuf[ymin:ymax+1, xmin:xmax+1])
        if not mask.any(): continue
        n = (w0[..., None]*VN[i0]+w1[..., None]*VN[i1]+w2[..., None]*VN[i2])
        n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-9)
        lam = 0.30+0.55*np.clip(n@L1, 0, 1)+0.25*np.clip(n@L2, 0, 1)
        col = np.clip(0.62*lam, 0, 1)
        sub = img[ymin:ymax+1, xmin:xmax+1]; sub[mask] = col[mask][..., None]
        zb = zbuf[ymin:ymax+1, xmin:xmax+1]; zb[mask] = zi[mask]
        fb = fid[ymin:ymax+1, xmin:xmax+1]; fb[mask] = fi
    return (img*255).astype(np.uint8), fid


def face_areas(m):
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    return 0.5*np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)


def handle_faces(m, nb=48, thresh=1.35, base_frac=0.15):
    """Vase body is a surface of revolution; the handle is the off-profile bulge.
    Bottom base_frac of the height is excluded so the SQUARE BASE's corners
    (also off-profile) are not mistaken for handle."""
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    y, r = V[:, 1], np.hypot(V[:, 0], V[:, 2])
    ylo, h = y.min(), y.max()-y.min()
    b = np.clip(((y-ylo)/h*nb).astype(int), 0, nb-1)
    prof = np.array([np.median(r[b == i]) if (b == i).any() else 0.0 for i in range(nb)])
    hv = (r > thresh*prof[b]+0.01) & (y > ylo+base_frac*h)
    return hv[F].all(1)


def _selftest(m):
    a = rf.render(m, 37.0, 13.0, res=256, dist=DIST, fov=FOV)
    b, _ = raster_ids(m, 37.0, 13.0, res=256)
    assert np.array_equal(a, b), "raster drifted from render_final.render()"
    print("selftest: raster identical to render_final.render()", flush=True)


def label(img, lines, px=384):
    t = Image.fromarray(np.array(Image.fromarray(img).resize((px, px))))
    d = ImageDraw.Draw(t)
    for i, s in enumerate(lines):
        d.text((6, 6+12*i), s, fill=(0, 0, 0))
    return np.array(t)


def sheet(tiles, ncol, path):
    while len(tiles) % ncol: tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i+ncol], 1) for i in range(0, len(tiles), ncol)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = rf.load_norm(MESH, 0)                       # y-up already; no prerot
    _selftest(m)
    A = face_areas(m); Atot = A.sum()
    HF = handle_faces(m); Ahand = A[HF].sum()
    print(f"mesh: {len(m.vertices)} V  {len(m.faces)} F | handle faces={HF.sum()} "
          f"({100*Ahand/Atot:.1f}% of area)", flush=True)
    rows = []

    def shoot(yaw, pitch, tag):
        img, fid = raster_ids(m, yaw, pitch)
        vis = np.unique(fid); vis = vis[vis >= 0]
        seen = np.zeros(len(A), bool); seen[vis] = True
        sil = float((fid >= 0).mean()*100)
        va = float(A[seen].sum()/Atot*100)
        ha = float(A[seen & HF].sum()/Ahand*100) if Ahand > 0 else float('nan')
        Image.fromarray(img).save(OUT/f"{tag}.png")
        rows.append((tag, yaw, pitch, sil, va, ha))
        print(f"{tag:>14}  yaw={yaw:5.1f} pitch={pitch:4.1f}  sil={sil:5.2f}%  "
              f"vis={va:5.2f}%  hand={ha:6.2f}%", flush=True)
        return label(img, [f"yaw {yaw:.0f}  pitch {pitch:.0f}",
                           f"vis {va:.1f}%  hand {ha:.1f}%"])

    print("--- yaw sweep @ pitch 10 ---", flush=True)
    yt = [shoot(y, 10.0, f"yaw_{int(y):03d}") for y in range(0, 360, 10)]
    sheet(yt, 6, OUT/"sheet_yaw.png")

    best = max([r for r in rows if r[2] == 10.0], key=lambda r: (r[4], r[5]))
    by = best[1]
    print(f"--- pitch sweep @ best yaw {by:.0f} ---", flush=True)
    pt = [shoot(by, p, f"pitch_{int(p):02d}") for p in (0, 5, 10, 15, 20, 25, 30, 40)]
    sheet(pt, 4, OUT/"sheet_pitch.png")

    with open(OUT/"metrics.tsv", "w") as fh:
        fh.write("tag\tyaw\tpitch\tsil_pct\tvis_area_pct\thandle_vis_pct\n")
        for r in rows:
            fh.write("%s\t%.1f\t%.1f\t%.4f\t%.4f\t%.4f\n" % r)

    top = max(rows, key=lambda r: (r[4], r[5]))
    print(f"\nBEST: yaw={top[1]:.0f} pitch={top[2]:.0f}  vis={top[4]:.2f}%  "
          f"hand={top[5]:.2f}%  ({top[0]}.png)", flush=True)


if __name__ == "__main__":
    main()
