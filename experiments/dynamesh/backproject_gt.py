"""
backproject_gt.py — lift the 150 GT video frames onto the fixed TRELLIS.2 mesh

WHY
  The loss currently compares our render against the raw video frame, but the two
  do not have the same silhouette. Measured over all 150 frames (largest
  connected component, min(RGB) < 245):

      GT area    102,549 -> 109,583 px   (+6.60%, monotone)
      GT height  328 -> 340 px           GT width 592 -> 596
      our render 28,559 px at 518^2, CONSTANT (mesh and camera are both fixed)

  So every frame carries a band where one image has object and the other has
  background, and a per-pixel loss charges the adapter for that shape difference
  as if it were texture error. rung13's alignment cannot remove it: that solved
  ONE static transform, and this mismatch grows with time.

  Backprojection moves supervision from image space onto the surface. Each
  surface point gets the colour of the video pixel it is actually seen at, so
  target and prediction share a silhouette by construction and the mismatch is
  resolved once, at projection time, instead of polluting every pixel.

HOW, AND THE TWO THINGS THAT MAKE IT CORRECT
  Pixel-driven, not vertex-driven. We rasterise the mesh from the confirmed
  camera and walk the VISIBLE fragments. A vertex-driven projection would assign
  a colour to back-facing points too — they project onto the silhouette and would
  receive front-surface colour, which is exactly the "sticker" artefact rung13
  diagnosed. Rasterisation gives occlusion for free: a fragment exists only where
  that surface is actually seen.

  INTERSECTION, not union. A pixel contributes only where the render is covered
  AND the GT is object. Pixels covered by our mesh but background in the video
  are the mismatch band; including them would paint white onto the surface.

WHAT IT WRITES
  targets_voxel  [F, N, 3]  float16   mean GT colour per latent voxel per frame
  counts_voxel   [F, N]     int32     contributing pixels (0 = no target)
  targets_vertex [F, V, 3]  float16   same, per mesh vertex (barycentric splat)
  counts_vertex  [F, V]     int32
  valid_voxel    [N]        bool      voxels with a target in EVERY frame
  a diagnostic render per --check-frames so the result can be looked at

  A voxel with count 0 has no supervision from this camera and never will — that
  is the coverage problem, stated per voxel, and it is left explicit rather than
  filled in.
"""

import argparse, json, math, os, sys, time
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
ap.add_argument('--resolution', type=int, default=512, choices=[512, 1024],
                help='TRELLIS.2 latent resolution; the latent grid is this / 16')
ap.add_argument('--raster-res', type=int, default=960,
                help='rasterise at the GT frame size so pixel i,j maps 1:1 and no '
                     'resampling is introduced between render and video')
ap.add_argument('--bg-thresh', type=int, default=245)
ap.add_argument('--check-frames', type=int, nargs='+', default=[1, 40, 75, 120, 150])
ap.add_argument('--tag', default='gt_backproj')
args = ap.parse_args()

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image
from scipy import ndimage

DEVICE = torch.device('cuda')
OUT = (_HERE / 'out' / args.tag).resolve()
(OUT / 'check').mkdir(parents=True, exist_ok=True)
LAT = args.resolution // 16

# ── camera: confirmed v1 front view (trellis/renderers/mesh_renderer.py) ─────
NEAR, FAR = 0.5, 3.0
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX, 0., 0.5], [0., _FX, 0.5], [0., 0., 1.]],
                          dtype=torch.float32)


def log(*a):
    print(*a, flush=True)


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    r = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    r[0, 0], r[1, 1] = 2 * fx, 2 * fy
    r[0, 2], r[1, 2] = 2 * cx - 1, -2 * cy + 1
    r[2, 2], r[2, 3] = far / (far - near), near * far / (near - far)
    r[3, 2] = 1.0
    return r


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline

    log('=' * 92)
    log('BACKPROJECT GT -> MESH SURFACE')
    log(f'  mesh {args.mesh}')
    log(f'  {args.n_frames} frames, rasterised at {args.raster_res}^2, latent grid {LAT}^3')
    log('=' * 92)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()

    mesh_in = trimesh.load(args.mesh, process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().to(DEVICE)
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    faces = torch.from_numpy(np.asarray(mesh_in.faces)).int().to(DEVICE).contiguous()
    V = v_raw.shape[0]

    shape_slat = pipe.encode_shape_slat(mesh_pp, args.resolution)
    coords = shape_slat.coords[:, 1:].long()
    N = coords.shape[0]
    key_c = (coords[:, 0] * LAT + coords[:, 1]) * LAT + coords[:, 2]
    order = torch.argsort(key_c)
    key_sorted = key_c[order]
    log(f'\n[MESH] {V:,} verts  {faces.shape[0]:,} faces   [LATENT] {N} voxels')

    # ── rasterise ONCE: mesh and camera are both fixed ───────────────────────
    R = args.raster_res
    full = (intrinsics_to_projection(INTRINSICS.to(DEVICE), NEAR, FAR)
            @ EXTRINSICS.to(DEVICE)).unsqueeze(0)
    vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
    clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, clip, faces, (R, R))          # [1,R,R,4]
    covered = rast[0, ..., 3] > 0                              # visible fragments
    tri_id = (rast[0, ..., 3].long() - 1)                      # triangle per pixel
    bary = rast[0, ..., :2]                                    # barycentric u,v

    # each covered pixel -> its 3D surface point in the PREPROCESSED frame,
    # which is the frame the latent voxels live in
    pos_pp = dr.interpolate(v_pp.unsqueeze(0).contiguous(), rast, faces)[0][0]  # [R,R,3]
    log(f'[RASTER] {int(covered.sum()):,} covered px of {R*R:,} '
        f'({100*float(covered.float().mean()):.1f}%)')

    # map every covered pixel to a latent voxel index (-1 if the voxel is not in
    # the sparse set, which can happen at the very edge of the surface)
    pix_idx = torch.nonzero(covered.reshape(-1)).squeeze(1)
    p = pos_pp.reshape(-1, 3)[pix_idx]
    vi = ((p + 0.5) * LAT).long().clamp(0, LAT - 1)
    key_p = (vi[:, 0] * LAT + vi[:, 1]) * LAT + vi[:, 2]
    slot = torch.searchsorted(key_sorted, key_p).clamp(max=N - 1)
    hit = key_sorted[slot] == key_p
    vox_of_pix = torch.where(hit, order[slot], torch.full_like(slot, -1))
    log(f'[MAP]    {int(hit.sum()):,} px land in a latent voxel '
        f'({100*float(hit.float().mean()):.1f}% of covered)')

    # each covered pixel also splats to its triangle's 3 vertices, weighted by
    # barycentric coordinate, so a per-VERTEX target is available too
    tri = tri_id.reshape(-1)[pix_idx]
    b = bary.reshape(-1, 2)[pix_idx]
    w3 = torch.stack([1.0 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], -1)   # [P,3]
    vert3 = faces.long()[tri]                                           # [P,3]

    gt_dir = Path(args.gt_dir)
    F = args.n_frames
    tv = np.zeros((F, N, 3), np.float16); cv = np.zeros((F, N), np.int32)
    tx = np.zeros((F, V, 3), np.float16); cx = np.zeros((F, V), np.int32)

    t0 = time.time()
    for i, fr in enumerate(range(1, F + 1)):
        img = np.array(Image.open(gt_dir / f'frame_{fr:04d}.png').convert('RGB')
                       .resize((R, R), Image.LANCZOS))
        gtm = largest_component(img.min(axis=2) < args.bg_thresh)
        gt = torch.from_numpy(img).float().to(DEVICE) / 255.0
        gtm_t = torch.from_numpy(gtm).to(DEVICE)

        # INTERSECTION: covered by the mesh AND object in the video. Pixels in the
        # mismatch band contribute nothing rather than painting background onto
        # the surface.
        keep = gtm_t.reshape(-1)[pix_idx]
        col = gt.reshape(-1, 3)[pix_idx]

        # ---- per voxel
        sel = keep & (vox_of_pix >= 0)
        acc = torch.zeros(N, 3, device=DEVICE)
        cnt = torch.zeros(N, device=DEVICE)
        acc.index_add_(0, vox_of_pix[sel], col[sel])
        cnt.index_add_(0, vox_of_pix[sel], torch.ones(int(sel.sum()), device=DEVICE))
        tv[i] = (acc / cnt.clamp_min(1).unsqueeze(-1)).cpu().numpy().astype(np.float16)
        cv[i] = cnt.cpu().numpy().astype(np.int32)

        # ---- per vertex (barycentric splat)
        accv = torch.zeros(V, 3, device=DEVICE)
        cntv = torch.zeros(V, device=DEVICE)
        for j in range(3):
            w = w3[:, j] * keep.float()
            accv.index_add_(0, vert3[:, j], col * w.unsqueeze(-1))
            cntv.index_add_(0, vert3[:, j], w)
        tx[i] = (accv / cntv.clamp_min(1e-6).unsqueeze(-1)).cpu().numpy().astype(np.float16)
        cx[i] = (cntv > 1e-6).cpu().numpy().astype(np.int32)

        if fr in args.check_frames:
            # paint the backprojected target back onto the surface and render it,
            # so the result can be inspected rather than trusted
            painted = torch.zeros(R * R, 3, device=DEVICE) + 1.0
            good = sel
            painted[pix_idx[good]] = col[good]
            Image.fromarray((painted.reshape(R, R, 3).cpu().numpy() * 255)
                            .astype(np.uint8)).save(OUT / 'check' / f'painted_f{fr:04d}.png')

        if fr % 10 == 0 or fr == 1:
            log(f'  {fr:3d}/{F}  voxels with target {int((cv[i] > 0).sum()):,}/{N}  '
                f'px used {int(sel.sum()):,}  {time.time()-t0:.0f}s')

    valid_v = (cv > 0).all(axis=0)
    valid_x = (cx > 0).all(axis=0)
    log(f'\n{"="*92}')
    log(f'[COVERAGE] voxels with a target in EVERY frame : {int(valid_v.sum()):,}/{N} '
        f'({100*valid_v.mean():.1f}%)')
    log(f'[COVERAGE] voxels with a target in NO frame    : {int((cv.sum(0) == 0).sum()):,}/{N}')
    log(f'[COVERAGE] vertices with a target in every frame: {int(valid_x.sum()):,}/{V} '
        f'({100*valid_x.mean():.1f}%)')
    log(f'[COVERAGE] the complement is the surface this camera never sees; it is')
    log(f'           left with count 0 rather than filled in.')
    log('=' * 92)

    np.savez_compressed(OUT / 'targets.npz',
                        targets_voxel=tv, counts_voxel=cv,
                        targets_vertex=tx, counts_vertex=cx,
                        valid_voxel=valid_v, valid_vertex=valid_x,
                        coords=coords.cpu().numpy())
    json.dump(dict(mesh=args.mesh, n_frames=F, resolution=args.resolution,
                   raster_res=R, latent_grid=LAT, n_voxels=int(N), n_vertices=int(V),
                   covered_px=int(covered.sum()),
                   voxels_every_frame=int(valid_v.sum()),
                   voxels_never=int((cv.sum(0) == 0).sum()),
                   vertices_every_frame=int(valid_x.sum())),
              open(OUT / 'backproj.json', 'w'), indent=2)
    log(f'[SAVE] {OUT}/targets.npz   check renders in {OUT}/check\n[DONE]')


if __name__ == '__main__':
    main()
