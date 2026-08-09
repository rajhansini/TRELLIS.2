"""
run_baseline_field.py — TRELLIS.2 per-frame baseline, rendered from the PBR FIELD

WHY THIS REPLACES THE UV-BAKED VERSION

  run_baseline.py rendered by letting postprocess_mesh() bake the generated PBR
  field into a UV atlas, then texture-sampling that atlas. The result came out
  dark, speckled and muddy — visibly worse than TRELLIS v1, which makes no sense
  for a newer model.

  It was not the model. It was the bake:

      mesh faces        430,910
      atlas texels    4,194,304   (2048 x 2048)
      texels per face         9.7   -- before xatlas packing margins

  A triangle needs roughly 16-32 texels to carry any detail. At <10 it aliases
  badly, which is exactly the speckle. The field was never the problem:
  diag_final.py sampled the SAME field directly at surface points for frame 75
  and produced clean lava cracks, and the round trip against the field's own
  voxel coordinates correlates at 0.9963.

  So this renders the way rung14 does — grid_sample_3d on the PBR voxel field at
  each pixel's 3D surface point. No atlas, no resolution ceiling, and it makes
  the baseline and rung14 pixel-comparable because they now share a render path.

TWO OTHER CONSEQUENCES, BOTH WANTED

  * shape_slat is encoded ONCE. It is a deterministic function of the fixed mesh
    (no noise anywhere in that stage), but pipe.run() recomputed it every frame —
    150 redundant o_voxel dual-grid builds. That was most of the 21 s/frame.
  * no UVs are needed, so the plain .ply is used rather than the 262k-vertex
    unwrapped .obj.

CAMERA
  The confirmed TRELLIS v1 front view, copied from trellis/renderers/
  mesh_renderer.py, so this is the same angle as every earlier rung and as
  rung14. GATE-cam checks it against the GT silhouette before rendering.
"""

import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', default='/net/projects/ranalab/rajhansini/TRELLIS/'
                                  'render/out/f0075/frozen_f0075.ply')
ap.add_argument('--gt-dir', default='/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                                    '/outputs/teapot_lava_kling_premium'
                                    '/teapot_lava_kling_premium_front/all_frames_150')
ap.add_argument('--n-frames', type=int, default=150)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--resolution', type=int, default=512, choices=[512, 1024])
ap.add_argument('--render-res', type=int, default=518)
ap.add_argument('--bg-thresh', type=int, default=245)
ap.add_argument('--tag', default='frozen_field')
ap.add_argument('--fps', type=int, default=15)
args = ap.parse_args()

OUT = (_HERE / 'out' / args.tag).resolve()
(OUT / 'renders').mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw


def log(*a):
    print(*a, flush=True)


def largest_component(mask):
    """Biggest blob only — one stray sub-threshold pixel at (959,0) otherwise
    stretches the union crop across the whole frame (teapot 102,778 px, next
    largest component 3 px)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


# ── camera: confirmed v1 front view (trellis/renderers/mesh_renderer.py) ─────
NEAR, FAR = 0.5, 3.0
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX, 0., 0.5], [0., _FX, 0.5], [0., 0., 1.]],
                          dtype=torch.float32)


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0], r[1, 1] = 2 * fx, 2 * fy
    r[0, 2], r[1, 2] = 2 * cx - 1, -2 * cy + 1
    r[2, 2], r[2, 3] = far / (far - near), near * far / (near - far)
    r[3, 2] = 1.0
    return r


class FieldRenderer:
    """
    Fixed mesh + fixed camera -> rasterise ONCE, then only sample per frame.

    v_raw is the mesh in the TRELLIS v1 frame (already registered to the video by
    export_frame_mesh.py) and drives rasterisation, so the image lines up with
    the GT exactly as in every earlier rung. v_pp is the same vertices after
    preprocess_mesh, interpolated across the raster to give each pixel its
    position in the frame the PBR field lives in. preprocess_mesh keeps vertex
    order and faces, which is what makes the pairing legal.
    """

    def __init__(self, v_raw, v_pp, faces, res, grid_res):
        import nvdiffrast.torch as dr
        self.dr, self.res, self.grid_res = dr, res, grid_res
        try:
            self.ctx = dr.RasterizeCudaContext()
        except Exception:
            self.ctx = dr.RasterizeGLContext()
        full = (intrinsics_to_projection(INTRINSICS.cuda(), NEAR, FAR)
                @ EXTRINSICS.cuda()).unsqueeze(0)
        vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
        self.v_clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        self.faces = faces.int().contiguous()
        self.rast, _ = dr.rasterize(self.ctx, self.v_clip, self.faces, (res, res))
        self.mask = self.rast[0, ..., 3] > 0
        self.idx = torch.nonzero(self.mask.reshape(-1)).squeeze(1)
        pos = dr.interpolate(v_pp.unsqueeze(0).contiguous(), self.rast, self.faces)[0]
        self.pos = pos[0].reshape(-1, 3)[self.idx].contiguous()
        self.n_px = int(self.mask.sum())

    def __call__(self, pbr):
        from flex_gemm.ops.grid_sample import grid_sample_3d
        attrs = grid_sample_3d(
            pbr.feats, pbr.coords,
            shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
            grid=((self.pos + 0.5) * self.grid_res).reshape(1, -1, 3),
            mode='trilinear')
        # channel axis is LAST; [:, :3] would slice the POINT axis and silently
        # mix in alpha (~1.0). That bug produced a whole afternoon of phantom
        # "the field is bright" readings.
        rgb = attrs[..., :3].reshape(-1, 3).float().clamp(0, 1)
        base = torch.ones(self.res * self.res, 3, device='cuda')
        return base.index_put((self.idx,), rgb).view(self.res, self.res, 3)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline

    log('=' * 88)
    log('TRELLIS.2 PER-FRAME BASELINE — rendered from the PBR FIELD (no UV bake)')
    log(f'  mesh {args.mesh}')
    log(f'  frames 1..{args.n_frames}   seed {args.seed} (FIXED)   res {args.resolution}')
    log('=' * 88)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{args.resolution}']
    dec = pipe.models['tex_slat_decoder']

    mesh_in = trimesh.load(args.mesh, process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    assert np.array_equal(np.asarray(mesh_pp.faces), np.asarray(mesh_in.faces))
    log(f'\n[MESH] {len(mesh_in.vertices):,} verts  {len(mesh_in.faces):,} faces  FIXED')

    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().cuda()
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().cuda()
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().cuda()
    rend = FieldRenderer(v_raw, v_pp, faces, args.render_res, args.resolution)
    log(f'[RENDER] v1 camera, {args.render_res}px, silhouette {rend.n_px:,} px, '
        f'rasterised once')

    gt_dir = Path(args.gt_dir)
    g75 = np.array(Image.open(gt_dir / 'frame_0075.png').convert('RGB')
                   .resize((args.render_res, args.render_res), Image.LANCZOS))
    gm = torch.from_numpy(g75.min(axis=2) < 242).cuda()
    iou = float((rend.mask & gm).sum() / max(int((rend.mask | gm).sum()), 1))
    log(f'[GATE-cam] silhouette IoU vs GT f75 = {iou:.4f}')
    assert iou > 0.80, f'GATE-cam FAILED ({iou:.4f}) — wrong camera or mesh frame'
    log('[GATE-cam] PASSED')

    # ── conditioning: deterministic, fixed union crop ────────────────────────
    frames = list(range(1, args.n_frames + 1))
    raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB'))
            for f in frames]
    alphas = [largest_component(r.min(axis=2) < args.bg_thresh) for r in raws]
    u = np.zeros_like(alphas[0])
    for a in alphas:
        u |= a
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))
    log(f'[COND] fixed union crop {bbox}, premultiplied to black (no BiRefNet)')

    # ── shape latent: ONCE. Deterministic, and the mesh never changes. ───────
    t0 = time.time()
    shape_slat = pipe.encode_shape_slat(mesh_pp, args.resolution)
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].cuda()
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].cuda()
    ss_n = (shape_slat - ss_mean) / ss_std
    log(f'[SHAPE] shape_slat {tuple(shape_slat.feats.shape)} in {time.time()-t0:.1f}s '
        f'— computed ONCE, reused for all {len(frames)} frames')

    renders, gts, per_frame = [], [], []
    for k, fi in enumerate(frames, 1):
        r, a = raws[k - 1], alphas[k - 1]
        rgba = np.concatenate([r, (a * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        cond_img = Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

        # no_grad: this is pure inference, and grid_sample_3d otherwise builds a
        # graph that both wastes memory and makes .numpy() raise.
        t1 = time.time()
        with torch.no_grad():
            torch.manual_seed(args.seed)      # FIXED across frames — charitable
            cond = pipe.get_cond([cond_img], args.resolution)
            slat = pipe.sample_tex_slat(cond, flow, shape_slat, {})
            pbr = pipe.decode_tex_slat(slat)
            img = rend(pbr).detach()
        dt = time.time() - t1

        arr = (img.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(OUT / 'renders' / f'frame_{fi:04d}.png')
        renders.append(arr)
        gts.append(np.array(Image.open(gt_dir / f'frame_{fi:04d}.png').convert('RGB')
                            .resize((args.render_res, args.render_res), Image.LANCZOS)))
        per_frame.append(dict(frame=fi, seconds=round(dt, 2)))
        del slat, pbr, img
        if k % 10 == 0 or k == 1:
            log(f'  {k:3d}/{len(frames)}  f{fi:04d}  {dt:.1f}s')
        torch.cuda.empty_cache()

    # ── temporal behaviour ───────────────────────────────────────────────────
    R = np.stack(renders).astype(np.float64)
    G = np.stack(gts).astype(np.float64)
    m = rend.mask.cpu().numpy()
    gmn = G[0].min(axis=2) < 245
    con = [float(np.abs(R[i] - R[i - 1])[m].mean()) for i in range(1, len(R))]
    dri = [float(np.abs(R[i] - R[0])[m].mean()) for i in range(1, len(R))]
    gcon = [float(np.abs(G[i] - G[i - 1])[gmn].mean()) for i in range(1, len(G))]
    gdri = [float(np.abs(G[i] - G[0])[gmn].mean()) for i in range(1, len(G))]

    log('\n' + '=' * 88)
    log('TEMPORAL BEHAVIOUR   (consecutive-frame change; lower = steadier)')
    log(f'{"":26}{"flicker":>14}{"drift":>12}{"flicker/drift":>16}')
    log('-' * 88)
    log(f'{"GT video":26}{np.mean(gcon):>14.3f}{gdri[-1]:>12.3f}'
        f'{np.mean(gcon)/max(gdri[-1],1e-9):>16.3f}')
    log(f'{"TRELLIS.2 (field)":26}{np.mean(con):>14.3f}{dri[-1]:>12.3f}'
        f'{np.mean(con)/max(dri[-1],1e-9):>16.3f}')
    log('=' * 88)
    json.dump(dict(mesh=args.mesh, n_frames=args.n_frames, seed=args.seed,
                   resolution=args.resolution, render_res=args.render_res,
                   gate_cam_iou=iou, crop_bbox=list(bbox),
                   flicker_mean=float(np.mean(con)), drift_final=dri[-1],
                   gt_flicker_mean=float(np.mean(gcon)), gt_drift_final=gdri[-1],
                   consecutive=con, drift=dri, per_frame=per_frame),
              open(OUT / 'baseline.json', 'w'), indent=2)

    # ── video ────────────────────────────────────────────────────────────────
    sd = OUT / 'strip'; sd.mkdir(exist_ok=True)
    LAB = 28
    for i, (g, r) in enumerate(zip(gts, renders), 1):
        H, W = r.shape[:2]
        canv = Image.fromarray(np.full((H + LAB, W * 2, 3), 18, np.uint8))
        d = ImageDraw.Draw(canv)
        for j, (a, lb) in enumerate([(g, 'GT video'),
                                     (r, 'TRELLIS.2 per-frame (frozen)')]):
            canv.paste(Image.fromarray(a.astype(np.uint8)), (j * W, LAB))
            d.rectangle([j * W, 0, (j + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
            d.text((j * W + 8, 8), lb, fill=(240, 240, 240))
        canv.save(sd / f'{i:04d}.png')
    vid = OUT / f'BASELINE_{args.tag}.mp4'
    pr = subprocess.run(['/usr/bin/ffmpeg', '-encoders'], capture_output=True, text=True)
    fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
          if 'libx264' in pr.stdout else
          ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
    subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(args.fps),
                    '-i', str(sd / '%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)],
                   check=True)
    log(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)\n[DONE]')


if __name__ == '__main__':
    main()
