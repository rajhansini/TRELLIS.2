"""
run_baseline.py — TRELLIS.2 per-frame texturing baseline
--------------------------------------------------------
ONE fixed mesh, N video frames, N independent texturing calls.

WHAT THIS IS
  TRELLIS.2 ships Trellis2TexturingPipeline: give it a mesh and ONE image, it
  returns that mesh with a generated PBR texture. The shape is an input and is
  never modified — which is exactly the setting we want (one mesh, a series of
  textures) and which TRELLIS v1 has no facility for at all.

  This script runs it once per video frame. That is the honest baseline for
  "dynamic texture from a video": the strongest thing you can do with the
  released model and no new training.

WHAT IT IS EXPECTED TO SHOW
  Each call is INDEPENDENT. Nothing ties frame t's texture to frame t-1's. So
  the prediction is temporal flicker, and quantifying it is the point of the run
  — that number is the problem statement for anything we build next.

  The seed is held FIXED across frames by default. That is deliberate and it is
  charitable to the baseline: it removes the random draw as a source of
  variation. Combined with the fixed mesh (shape_slat is a deterministic
  encode, no noise), EVERY frame-to-frame difference that remains enters
  through the DINOv3 image conditioning -> cross-attention. That makes the
  flicker number a clean measurement of one pathway.
  --vary-seed flips this to the pessimistic case for comparison.

THREE CONTROLS THAT MAKE THE NUMBER MEAN SOMETHING
  1. UV LAYOUT IS PINNED.  postprocess_mesh() reuses the input mesh's UVs if it
     has any, and only falls back to cumesh.uv_unwrap() when it does not. A bare
     .ply has none, so the unwrap would re-run per frame and each frame's
     texture map could land in a different layout — making the maps mutually
     incomparable. We unwrap ONCE up front and feed the UV'd mesh to every
     frame, so all N texture maps share one layout and can be diffed texel by
     texel. --no-preunwrap restores the old behaviour.

  2. REMBG IS OUT OF THE LOOP.  pipe.run(preprocess_image=True) would run
     BiRefNet on every frame; its mask jitters frame to frame and that jitter
     would be counted as texture flicker. We derive alpha deterministically
     from the near-white background instead, crop with a FIXED union bbox
     computed once over all frames, and pass preprocess_image=False. Framing is
     then bit-identical across frames and contributes exactly zero.

  3. RENDERING DOES NOT DEPEND ON pyglet.  trimesh's scene.save_image() needs
     pyglet, which is not installed in this env — it silently returned None for
     every frame, leaving no metric and no video. Rendering is done with
     nvdiffrast (same dependency the pipeline itself already uses).

WHAT IT MEASURES
  Two independent views of the same quantity:

    UV-SPACE   mean |T_t - T_{t-1}| over valid texels of the base-colour map.
               Exact, renderer-free, and only meaningful because of control 1.
    IMAGE-SPACE  the same on a fixed-camera render of the textured mesh.

  For each, flicker (consecutive change) is reported against drift (change from
  frame 1). A texture that legitimately EVOLVES has low flicker and high drift.
  One that jitters in place has high flicker and low drift.

  Flicker is a CONSECUTIVE-frame quantity, so frames are processed at stride 1.
  Subsampling would measure multi-frame drift instead and is not the same thing.

OUTPUTS
  out/<tag>/renders/frame_####.png      fixed-camera render of each frame
  out/<tag>/texmaps/frame_####.jpg      base-colour UV map (downsampled preview)
  out/<tag>/textured/frame_####.glb     the textured mesh (only with --keep-glb)
  out/<tag>/baseline.json               per-frame and aggregate metrics
  out/<tag>/BASELINE_<tag>.mp4          GT | baseline, side by side

Usage (must run on a compute node — o_voxel/cumesh/nvdiffrast need GLIBC 2.32):
  python run_baseline.py --mesh mesh.ply --n-frames 150
  python run_baseline.py --mesh mesh.ply --n-frames 8 --tag smoke   # quick check
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME']              = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

TRELLIS2_DIR = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
sys.path.insert(0, str(TRELLIS2_DIR))

_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True, help='the ONE mesh to texture, .ply or .glb')
ap.add_argument('--gt-dir', default='/net/projects/ranalab/rajhansini/MV-Adapter-Experimental'
                                    '/outputs/teapot_lava_kling_premium'
                                    '/teapot_lava_kling_premium_front/all_frames_150')
ap.add_argument('--n-frames',   type=int, default=150)
ap.add_argument('--seed',       type=int, default=42)
ap.add_argument('--vary-seed',  action='store_true',
                help='use seed=frame index instead of a fixed seed (pessimistic case)')
ap.add_argument('--resolution', type=int, default=1024, choices=[512, 1024])
ap.add_argument('--texture-size', type=int, default=2048)
ap.add_argument('--tag',        default='frozen_t2')
ap.add_argument('--out-dir',    default=None, type=Path)
ap.add_argument('--fps',        type=int, default=15)
ap.add_argument('--render-res', type=int, default=512)
ap.add_argument('--bg-thresh',  type=int, default=245,
                help='pixels with min(RGB) >= this are background in the GT frames')
ap.add_argument('--no-preunwrap', action='store_true',
                help='do NOT unwrap once up front; let the pipeline unwrap per frame '
                     '(old behaviour — texture maps are then not comparable)')
ap.add_argument('--keep-glb',   action='store_true',
                help='keep every per-frame .glb (150 x ~10-40 MB). Off by default.')
args = ap.parse_args()

OUT = (args.out_dir or (_HERE / 'out' / args.tag)).resolve()
(OUT / 'renders').mkdir(parents=True, exist_ok=True)
(OUT / 'texmaps').mkdir(parents=True, exist_ok=True)
if args.keep_glb:
    (OUT / 'textured').mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import trimesh
from PIL import Image


# ── trimesh: hand back WRITABLE vertex normals ───────────────────────────────
# postprocess_mesh() flips axes in place for GLB convention:
#     trellis2_texturing.py:362-363
#     vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
#     normals[:, 1],  normals[:, 2]  = normals[:, 2],  -normals[:, 1]
# mesh.vertices is writable, but mesh.vertex_normals is a cached TrackedArray with
# writeable=False, so the second line raises
#     ValueError: assignment destination is read-only
# Only the UV-SUPPLIED branch reaches it: without UVs the code runs
# `normals = normals[vmap]` (:316), whose fancy-indexing result is a writable copy
# by accident. Supplying UVs — which is the whole point of pinning one atlas
# across all 150 frames — skips that line and exposes the bug.
#
# Reassigning the attribute does not help; trimesh re-wraps it read-only. So the
# property itself is patched to return a fresh writable copy. This patches
# trimesh, a third-party library, from our own script — TRELLIS.2's source is
# still never modified.
_tm_vn = trimesh.Trimesh.vertex_normals
trimesh.Trimesh.vertex_normals = property(
    lambda self: np.array(_tm_vn.fget(self), dtype=np.float64),
    _tm_vn.fset, _tm_vn.fdel)


def log(*a):
    print(*a, flush=True)


def largest_component(mask):
    """
    Keep only the biggest blob of a foreground mask.

    The GT frames carry a few stray dark pixels — frame 4 has ONE at (959, 0)
    valued [243,245,242], just under the 245 threshold. A raw min/max bbox over
    the union of all frames then spans the whole image: measured (0, 238, 802,
    1040) instead of the correct (212, 188, 802, 778), which both pads outside
    the 960px frame and shrinks the teapot inside the conditioning crop.
    Frame 4 has 89 connected components: the teapot at 102,778 px and the next
    largest at 3 px. Taking the biggest component is therefore decisive, not a
    tuned threshold.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)



# ─────────────────────────────────────────────────────────────────────────────
# CONTROL 2 — deterministic conditioning images, rembg never runs at inference
# ─────────────────────────────────────────────────────────────────────────────
def build_cond_images(gt_dir: Path, frames, thresh: int):
    """
    Replicate Trellis2TexturingPipeline.preprocess_image() deterministically.

    The pipeline would run BiRefNet per frame and crop to that frame's own alpha
    bbox. Both are per-frame neural/data-dependent steps, and both inject
    frame-to-frame variation that is not texture. Here alpha comes from the
    near-white background by a fixed threshold, and the crop box is the UNION
    over all frames — computed once, applied identically to every frame.

    Returns (cond_images, gt_rgb_arrays, bbox).
    """
    raws = []
    for fi in frames:
        im = Image.open(gt_dir / f'frame_{fi:04d}.png').convert('RGB')
        raws.append(np.array(im))

    # alpha: foreground is anything darker than the off-white background
    alphas = [largest_component(r.min(axis=2) < thresh) for r in raws]

    # union bbox over every frame, then squared and centred like the pipeline does
    union = np.zeros_like(alphas[0])
    for a in alphas:
        union |= a
    ys, xs = np.where(union)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0))
    bbox = (int(cx - size // 2), int(cy - size // 2),
            int(cx + size // 2), int(cy + size // 2))

    cond_images = []
    for r, a in zip(raws, alphas):
        rgba = np.concatenate([r, (a * 255).astype(np.uint8)[..., None]], axis=-1)
        crop = Image.fromarray(rgba).crop(bbox)
        f = np.asarray(crop).astype(np.float32) / 255.0
        # premultiply by alpha -> black background, exactly as the pipeline does
        cond_images.append(Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255)
                                           .astype(np.uint8)))
    return cond_images, raws, bbox


# ─────────────────────────────────────────────────────────────────────────────
# CONTROL 1 — pin the UV layout once so every frame's texture map is comparable
# ─────────────────────────────────────────────────────────────────────────────
def preunwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    UV-unwrap the mesh a single time and attach the UVs.

    postprocess_mesh() reuses mesh.visual.uv when it exists and only calls
    cumesh.uv_unwrap() when it does not. Doing it here means all N frames bake
    into ONE layout, so the resulting texture maps can be differenced texel by
    texel. It also fixes the vertex count: unwrapping splits seams, so the
    returned mesh has more vertices than the input, and doing it per frame made
    the output count disagree with the input for a reason unrelated to geometry.

    xatlas, not cumesh. cumesh's compiled kernels raise CUDA error 209 ("no
    kernel image is available for execution on the device") on this cluster's
    A40s — it was built for a different arch. That failure is not avoidable by
    skipping this function, because postprocess_mesh() calls the same
    cumesh.uv_unwrap() internally whenever the mesh arrives without UVs.
    Supplying UVs here is therefore both the fix and the control: cumesh is
    never reached at all. xatlas runs on CPU, so no arch can bite us.
    """
    if getattr(getattr(mesh, 'visual', None), 'uv', None) is not None:
        log('[UV] mesh already carries UVs — reusing them, no unwrap needed')
        return mesh

    import xatlas
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    vmap, idx, uv = xatlas.parametrize(v, f)

    out = trimesh.Trimesh(vertices=v[vmap].astype(np.float64),
                          faces=idx.astype(np.int64), process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=uv.astype(np.float64), material=trimesh.visual.material.PBRMaterial())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CONTROL 3 — nvdiffrast renderer (trimesh's needs pyglet, which is absent here)
# ─────────────────────────────────────────────────────────────────────────────
# ── camera: the CONFIRMED TRELLIS v1 view, not a fitted one ──────────────────
# The previous renderer fitted a camera to the mesh's own bounds looking down +Z
# and applied it to the mesh returned by postprocess_mesh — which is in the GLB
# frame (axis-swapped at trellis2_texturing.py:362-363). That is an arbitrary
# orientation with no relationship to the GT video's viewpoint, so GT|ours strips
# compared two different views of the object and every flicker/drift number was
# measured on a render that does not correspond to the video.
#
# rung14 renders through the v1 EXTRINSICS/INTRINSICS and scores silhouette IoU
# 0.9052 against the GT. The baseline must use the SAME camera or the two are not
# comparable. Constants and the projection are copied verbatim from
# trellis/renderers/mesh_renderer.py so nothing is re-derived.
RENDER_NEAR, RENDER_FAR = 0.5, 3.0
_FX_N = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXTRINSICS = torch.tensor([[1., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 2.],
                           [0., 0., 0., 1.]], dtype=torch.float32)
INTRINSICS = torch.tensor([[_FX_N, 0., 0.5],
                           [0., _FX_N, 0.5],
                           [0., 0., 1.]], dtype=torch.float32)


def intrinsics_to_projection(intr, near, far):
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ret = torch.zeros((4, 4), dtype=intr.dtype, device=intr.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class V1CameraRenderer:
    """
    Render the textured mesh from the SAME fixed view rung14 and every earlier
    rung used, so the baseline and rung14 are directly comparable.

    TWO FRAMES, exactly as in rung14's SurfaceRenderer:
      the INPUT mesh is in the TRELLIS v1 frame and is already registered to the
      video (export_frame_mesh.py applied the alignment before writing the .ply),
      so it is what we RASTERISE.
      the OUTPUT mesh from postprocess_mesh carries the generated UVs and texture
      but has been through preprocess_mesh's normalisation AND the GLB axis swap,
      so its vertex POSITIONS are useless to us — we take only its uv + texture.

    That pairing is only valid because postprocess_mesh preserves vertex order and
    faces when UVs are supplied (trellis2_texturing.py:296-304 reuses them; only
    the no-UV branch calls uv_unwrap and reindexes). Asserted at run time.
    """

    def __init__(self, v_raw, faces, res):
        import nvdiffrast.torch as dr
        self.dr = dr
        try:
            self.ctx = dr.RasterizeCudaContext(); self.kind = 'cuda'
        except Exception:
            self.ctx = dr.RasterizeGLContext(); self.kind = 'gl'
        self.res = res
        ext, intr = EXTRINSICS.cuda(), INTRINSICS.cuda()
        full = (intrinsics_to_projection(intr, RENDER_NEAR, RENDER_FAR) @ ext).unsqueeze(0)
        v = v_raw.unsqueeze(0)
        vh = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        self.v_clip = torch.bmm(vh, full.transpose(-1, -2)).contiguous()
        self.faces = faces.int().contiguous()
        self.rast, _ = dr.rasterize(self.ctx, self.v_clip, self.faces, (res, res))
        self.mask = (self.rast[0, ..., 3] > 0)
        self.n_px = int(self.mask.sum())

    def __call__(self, out_mesh) -> np.ndarray:
        dr = self.dr
        uv = torch.from_numpy(np.asarray(out_mesh.visual.uv)).float().cuda()
        tex = np.asarray(out_mesh.visual.material.baseColorTexture.convert('RGB'))
        tex = torch.from_numpy(tex).float().cuda() / 255.0
        uvi, _ = dr.interpolate(uv.unsqueeze(0).contiguous(), self.rast, self.faces)
        col = dr.texture(tex.unsqueeze(0), uvi, filter_mode='linear')
        m = (self.rast[..., 3:] > 0).float()
        img = col * m + (1.0 - m)
        img = dr.antialias(img, self.rast, self.v_clip, self.faces)
        return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def flicker_stats(seq, mask):
    """consecutive-frame change vs change-from-frame-1, over `mask`."""
    consec = [float(np.abs(seq[i].astype(np.float64) - seq[i - 1].astype(np.float64))[mask].mean())
              for i in range(1, len(seq))]
    drift = [float(np.abs(seq[i].astype(np.float64) - seq[0].astype(np.float64))[mask].mean())
             for i in range(1, len(seq))]
    return consec, drift


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline

    log('=' * 88)
    log('TRELLIS.2 PER-FRAME TEXTURING BASELINE')
    log(f'  mesh       : {args.mesh}')
    log(f'  frames     : 1..{args.n_frames}  (stride 1 — flicker is a consecutive-frame quantity)')
    log(f'  seed       : {"frame index (VARYING)" if args.vary_seed else f"{args.seed} (FIXED — charitable to the baseline)"}')
    log(f'  resolution : {args.resolution}   texture: {args.texture_size}')
    log(f'  out        : {OUT}')
    log('=' * 88)

    mesh_in = trimesh.load(args.mesh, process=False, force='mesh')
    log(f'\n[MESH] {len(mesh_in.vertices):,} vertices  {len(mesh_in.faces):,} faces')
    b = mesh_in.bounds
    log(f'       bounds min {np.round(b[0], 4).tolist()}  max {np.round(b[1], 4).tolist()}')
    log(f'       THIS MESH IS FIXED — it is an input and the pipeline never modifies it.')

    # ── CONTROL 1 ────────────────────────────────────────────────────────────
    if args.no_preunwrap:
        log('\n[UV] --no-preunwrap: the pipeline will unwrap per frame.')
        log('     Texture maps across frames are NOT guaranteed to share a layout.')
    else:
        t0 = time.time()
        mesh_in = preunwrap(mesh_in)
        log(f'\n[UV] pre-unwrapped once in {time.time()-t0:.1f}s  -> '
            f'{len(mesh_in.vertices):,} vertices  {len(mesh_in.faces):,} faces')
        log(f'     (vertex count rises because unwrapping splits UV seams; geometry is identical)')
        log(f'     every frame now bakes into THIS layout, so the maps are comparable texel-wise')

    # ── CONTROL 2 ────────────────────────────────────────────────────────────
    gt_dir = Path(args.gt_dir)
    frames = list(range(1, args.n_frames + 1))
    t0 = time.time()
    cond_images, gt_raw, bbox = build_cond_images(gt_dir, frames, args.bg_thresh)
    log(f'\n[COND] {len(cond_images)} conditioning images built in {time.time()-t0:.1f}s')
    log(f'       alpha from background threshold {args.bg_thresh}, NOT BiRefNet')
    log(f'       fixed union crop bbox {bbox} -> identical framing every frame')
    log(f'       pipeline called with preprocess_image=False, so rembg never runs')

    log('\n[LOAD] Trellis2TexturingPipeline from microsoft/TRELLIS.2-4B ...')
    t0 = time.time()
    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.cuda()
    log(f'[LOAD] done in {time.time()-t0:.1f}s')

    v_raw = torch.from_numpy(np.asarray(mesh_in.vertices)).float().cuda()
    faces_raw = torch.from_numpy(np.asarray(mesh_in.faces)).int().cuda()
    renderer = V1CameraRenderer(v_raw, faces_raw, args.render_res)
    log(f'[RENDER] nvdiffrast {renderer.kind}, {args.render_res}px, CONFIRMED v1 camera '
        f'(same view as rung14 and every earlier rung)')

    # ── GATE-cam: the view must match the video, or nothing below is comparable.
    # The old fitted camera silently rendered an arbitrary orientation; this makes
    # that failure impossible to miss. rung14 measures 0.9052 here.
    _g = np.array(Image.open(gt_dir / 'frame_0075.png').convert('RGB')
                  .resize((args.render_res, args.render_res), Image.LANCZOS))
    _gm = torch.from_numpy(_g.min(axis=2) < 242).cuda()
    _iou = float((renderer.mask & _gm).sum() / max(int((renderer.mask | _gm).sum()), 1))
    log(f'[GATE-cam] silhouette IoU vs GT frame 75 = {_iou:.4f}  '
        f'(render {renderer.n_px:,} px, GT {int(_gm.sum()):,} px)')
    assert _iou > 0.80, (
        f'GATE-cam FAILED: IoU {_iou:.4f}. The camera or the mesh frame is wrong; '
        f'rung14 measures 0.9052 with the same mesh and camera.')
    log('[GATE-cam] PASSED')

    renders, gts, per_frame = [], [], []
    tex_prev, tex_first = None, None
    tex_consec, tex_drift = [], []
    verts_ref = len(mesh_in.vertices)

    for k, fi in enumerate(frames, 1):
        seed = fi if args.vary_seed else args.seed

        t1 = time.time()
        out_mesh = pipe.run(
            mesh_in, cond_images[k - 1], seed=seed,
            preprocess_image=False,
            resolution=args.resolution, texture_size=args.texture_size)
        dt = time.time() - t1

        nv = len(out_mesh.vertices)

        # ── UV-space flicker: exact, renderer-free ───────────────────────────
        tex = np.asarray(out_mesh.visual.material.baseColorTexture.convert('RGB'))
        if tex_first is None:
            tex_first = tex.copy()
            tex_valid = tex_first.min(axis=2) < 255           # texels the atlas uses
        else:
            tex_consec.append(float(np.abs(tex.astype(np.float64)
                                           - tex_prev.astype(np.float64))[tex_valid].mean()))
            tex_drift.append(float(np.abs(tex.astype(np.float64)
                                          - tex_first.astype(np.float64))[tex_valid].mean()))
        tex_prev = tex
        Image.fromarray(tex).resize((512, 512), Image.LANCZOS).save(
            OUT / 'texmaps' / f'frame_{fi:04d}.jpg', quality=92)

        if args.keep_glb:
            out_mesh.export(str(OUT / 'textured' / f'frame_{fi:04d}.glb'),
                            extension_webp=True)

        # ── image-space ──────────────────────────────────────────────────────
        # postprocess_mesh must not have reindexed: we rasterise the INPUT mesh
        # and interpolate the OUTPUT mesh's uv, so the two must share vertex order.
        assert len(out_mesh.visual.uv) == len(mesh_in.vertices), (
            f'vertex count changed ({len(mesh_in.vertices)} -> {len(out_mesh.visual.uv)}); '
            f'the uv/raster pairing is invalid. Was the mesh fed without UVs?')
        img_r = renderer(out_mesh)
        Image.fromarray(img_r).save(OUT / 'renders' / f'frame_{fi:04d}.png')
        renders.append(img_r)
        gts.append(np.array(Image.fromarray(gt_raw[k - 1])
                            .resize((args.render_res, args.render_res), Image.LANCZOS)))

        per_frame.append(dict(frame=fi, seed=seed, verts=nv, seconds=dt))
        if k % 10 == 0 or k == 1 or k == len(frames):
            log(f'  {k:3d}/{len(frames)}  f{fi:04d}  verts={nv:,}  {dt:.1f}s')

    # ---- geometry check -----------------------------------------------------
    vs = np.array([r['verts'] for r in per_frame])
    log(f'\n[GEOMETRY] vertex count across frames: min={vs.min():,} max={vs.max():,}')
    log(f'           input mesh had {verts_ref:,}')
    same = (vs.min() == vs.max() == verts_ref)
    log(f'           -> {"CONSTANT and equal to the input — one mesh confirmed" if same else "CONSTANT across frames" if vs.min()==vs.max() else "VARIES — investigate"}')

    # ---- flicker ------------------------------------------------------------
    metrics = {}
    if len(renders) >= 2:
        R = np.stack(renders)
        m = R[0].min(axis=2) < 245
        consec, drift = flicker_stats(R, m)
        G = np.stack(gts)
        gm = G[0].min(axis=2) < 245
        g_consec, g_drift = flicker_stats(G, gm)

        metrics = dict(
            flicker_mean=float(np.mean(consec)), flicker_max=float(np.max(consec)),
            drift_final=float(drift[-1]),
            gt_flicker_mean=float(np.mean(g_consec)), gt_drift_final=float(g_drift[-1]),
            flicker_over_drift=float(np.mean(consec) / max(drift[-1], 1e-9)),
            gt_flicker_over_drift=float(np.mean(g_consec) / max(g_drift[-1], 1e-9)),
            consecutive=consec, drift=drift,
            uv_flicker_mean=float(np.mean(tex_consec)) if tex_consec else None,
            uv_drift_final=float(tex_drift[-1]) if tex_drift else None,
            uv_flicker_over_drift=(float(np.mean(tex_consec) / max(tex_drift[-1], 1e-9))
                                   if tex_drift else None),
            uv_consecutive=tex_consec, uv_drift=tex_drift)

        log('\n' + '=' * 88)
        log('TEMPORAL BEHAVIOUR   (consecutive-frame change; lower = steadier)')
        log(f'{"":26}{"flicker (L1/frame)":>22}{"drift by end":>16}{"flicker/drift":>16}')
        log('-' * 88)
        log(f'{"GT video":26}{np.mean(g_consec):>22.3f}{g_drift[-1]:>16.3f}'
            f'{np.mean(g_consec)/max(g_drift[-1],1e-9):>16.3f}')
        log(f'{"TRELLIS.2 (image space)":26}{np.mean(consec):>22.3f}{drift[-1]:>16.3f}'
            f'{np.mean(consec)/max(drift[-1],1e-9):>16.3f}')
        if tex_consec:
            log(f'{"TRELLIS.2 (UV space)":26}{np.mean(tex_consec):>22.3f}{tex_drift[-1]:>16.3f}'
                f'{np.mean(tex_consec)/max(tex_drift[-1],1e-9):>16.3f}')
        log('-' * 88)
        ratio = np.mean(consec) / max(np.mean(g_consec), 1e-9)
        log(f'  baseline flickers {ratio:.2f}x the GT video')
        log('  A texture that legitimately EVOLVES has low flicker and high drift.')
        log('  High flicker with low drift = jittering in place, not evolving.')
        log('=' * 88)

    json.dump(dict(mesh=str(args.mesh), n_frames=args.n_frames,
                   seed=('varying' if args.vary_seed else args.seed),
                   resolution=args.resolution, texture_size=args.texture_size,
                   preunwrapped=(not args.no_preunwrap), crop_bbox=list(bbox),
                   input_vertices=verts_ref, per_frame=per_frame, **metrics),
              open(OUT / 'baseline.json', 'w'), indent=2)

    # ---- video --------------------------------------------------------------
    if len(renders) >= 2:
        strip_dir = OUT / 'strip'; strip_dir.mkdir(exist_ok=True)
        from PIL import ImageDraw
        LAB = 28
        for i, (g, r) in enumerate(zip(gts, renders), 1):
            H, W = r.shape[:2]
            canv = Image.fromarray(np.full((H + LAB, W * 2, 3), 18, np.uint8))
            d = ImageDraw.Draw(canv)
            for j, (a, lb) in enumerate([(g, 'GT video'),
                                         (r, 'TRELLIS.2 per-frame (frozen mesh)')]):
                canv.paste(Image.fromarray(a.astype(np.uint8)), (j * W, LAB))
                d.rectangle([j * W, 0, (j + 1) * W - 1, LAB - 1], fill=(38, 38, 58))
                d.text((j * W + 8, 8), lb, fill=(240, 240, 240))
            canv.save(strip_dir / f'{i:04d}.png')
        vid = OUT / f'BASELINE_{args.tag}.mp4'
        pr = subprocess.run(['/usr/bin/ffmpeg', '-encoders'],
                            capture_output=True, text=True)
        fl = (['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
              if 'libx264' in pr.stdout else
              ['-c:v', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p'])
        subprocess.run(['/usr/bin/ffmpeg', '-y', '-framerate', str(args.fps),
                        '-i', str(strip_dir / '%04d.png'),
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', *fl, str(vid)],
                       check=True)
        log(f'\n[VIDEO] {vid}  ({vid.stat().st_size/1e6:.1f} MB)')

    log(f'\n[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
