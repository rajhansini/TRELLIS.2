"""
find_orientation.py — which way round should the mesh face?

Silhouette IoU cannot answer this for a near-symmetric object: a cow's outline
from the front and the back are almost the same. So this renders the mesh
NORMAL-SHADED at a sweep of yaw angles and scores each against the (untextured)
first video frame on shading, not on outline.

Writes a contact sheet so the choice can be eyeballed as well as scored.
"""
import argparse, math
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--frame', required=True)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--yaws', default='0,45,90,135,180,225,270,315')
ap.add_argument('--tag', default='orientation')
A = ap.parse_args()

DEVICE = 'cuda'
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.], [0., 0., -1., 0.],
                           [0., 1., 0., 2.], [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5], [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)
NEAR, FAR = 0.5, 3.0


def i2p(intr, near, far):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0] = 2 * fx; r[1, 1] = 2 * fy
    r[0, 2] = 2 * cx - 1; r[1, 2] = -2 * cy + 1
    r[2, 2] = far / (far - near); r[2, 3] = near * far / (near - far); r[3, 2] = 1.
    return r


def normalise(v):
    vmin, vmax = v.min(0), v.max(0)
    v = (v - (vmin + vmax) / 2) * (0.99999 / (vmax - vmin).max())
    t = v[:, 1].copy(); v[:, 1] = -v[:, 2]; v[:, 2] = t
    return v


def yaw_about_up(v, deg):
    """Rotate about the camera's UP axis (z after the swap) — turns the object round."""
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return v @ R.T


def main():
    import nvdiffrast.torch as dr
    m = trimesh.load(A.mesh, process=False, force='mesh')
    V0 = normalise(np.asarray(m.vertices, np.float64).copy())
    F = np.asarray(m.faces)
    ctx = dr.RasterizeCudaContext()
    f_t = torch.from_numpy(F).int().to(DEVICE).contiguous()
    full = (i2p(INTRINSICS.to(DEVICE), NEAR, FAR) @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    eye = -(EXTRINSICS[:3, :3].T @ EXTRINSICS[:3, 3]).to(DEVICE)

    gt = np.asarray(Image.open(A.frame).convert('RGB'), np.float32) / 255
    if gt.shape[0] != A.res:
        gt = np.asarray(Image.open(A.frame).convert('RGB').resize((A.res, A.res),
                        Image.LANCZOS), np.float32) / 255
    gt_g = gt.mean(2)
    gt_m = gt.min(2) < 0.95

    def shade(v_np):
        mesh = trimesh.Trimesh(vertices=v_np, faces=F, process=False)
        vn = np.asarray(mesh.vertex_normals)
        v = torch.from_numpy(v_np).float().to(DEVICE)
        vn_t = torch.from_numpy(vn).float().to(DEVICE)
        vh = torch.cat([v, torch.ones_like(v[:, :1])], -1).unsqueeze(0)
        clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        rast, _ = dr.rasterize(ctx, clip, f_t, (A.res, A.res))
        mask = (rast[0, ..., 3] > 0)
        n = dr.interpolate(vn_t.unsqueeze(0).contiguous(), rast, f_t)[0][0]
        n = torch.nn.functional.normalize(n, dim=-1)
        p = dr.interpolate(v.unsqueeze(0).contiguous(), rast, f_t)[0][0]
        vd = torch.nn.functional.normalize(eye - p, dim=-1)
        lam = (n * vd).sum(-1).clamp(0, 1)
        img = torch.ones(A.res, A.res, device=DEVICE)
        img[mask] = (0.25 + 0.75 * lam)[mask]        # simple headlight shading
        return img.cpu().numpy(), mask.cpu().numpy()

    yaws = [float(x) for x in A.yaws.split(',')]
    out = _HERE / 'out' / A.tag; out.mkdir(parents=True, exist_ok=True)
    tiles, rows = [], []
    print(f'{"yaw":>6} {"silhouette IoU":>15} {"shading corr":>13}   (higher is better)')
    for y in yaws:
        g, msk = shade(yaw_about_up(V0.copy(), y))
        inter = np.logical_and(msk, gt_m).sum(); union = np.logical_or(msk, gt_m).sum()
        iou = inter / max(union, 1)
        both = msk & gt_m
        if both.sum() > 100:
            a = g[both] - g[both].mean(); b = gt_g[both] - gt_g[both].mean()
            corr = float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))
        else:
            corr = 0.0
        rows.append((corr, iou, y))
        print(f'{y:>6.0f} {iou:>15.4f} {corr:>13.4f}')
        tile = (g * 255).astype(np.uint8)
        Image.fromarray(tile).save(out / f'yaw_{int(y):03d}.png')
        tiles.append(tile)

    n = len(tiles); W = A.res
    sheet = Image.new('L', (W * n, W), 255)
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (i * W, 0))
    sheet.save(out / 'contact_sheet.png')
    Image.fromarray((gt_g * 255).astype(np.uint8)).save(out / 'video_frame.png')

    rows.sort(reverse=True)
    print(f'\n[BEST by shading correlation] yaw {rows[0][2]:.0f}  '
          f'corr {rows[0][0]:.4f}  IoU {rows[0][1]:.4f}')
    print(f'  contact sheet: {out}/contact_sheet.png   (yaws left to right: {A.yaws})')
    print(f'  video frame  : {out}/video_frame.png')


if __name__ == '__main__':
    main()
