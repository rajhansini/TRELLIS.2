"""
diag_final.py — vertices vs interpolated pixels, ONE field, ONE call each.

NOTE the slicing. grid_sample_3d returns [1, N, C]; the channel axis is LAST.
Earlier diagnostics used [:, :3], which slices the POINT axis — it returned the
first three points with ALL SIX channels, so alpha (~1.0) dominated and every
"bright" reading (round trip 0.4063, vertices 0.4037, the whole side split) was
an artefact. postprocess_mesh indexes attrs[..., 0:3] and SurfaceRenderer does
the same; only these probes were wrong.

Contradictions to resolve, all measured today:
  * field's own stored base_color mean = 0.0917
  * sampling that field AT ITS OWN voxel coords = 0.4063   <-- must equal the above
  * sampling at mesh vertices                   = 0.4037
  * sampling at interpolated pixel positions    = 0.0364   <-- the renderer
Every half of the object is bright at vertices (0.37-0.40), so it is not a
front/back issue. Either the vertex numbers are an artefact of a grid convention
I have wrong, or the interpolated positions differ from vertices in some way the
bounds checks did not reveal.

This samples ONE pbr field with the SAME call at: its own coords, all vertices,
only camera-visible vertices, and the renderer's interpolated pixel positions —
then prints a direct per-voxel round-trip error, which is the ground truth for
whether the convention is right at all.
"""
import os, sys, math
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
import numpy as np, torch, trimesh
import nvdiffrast.torch as dr
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline
from flex_gemm.ops.grid_sample import grid_sample_3d

RES, IMG = 512, 512
R = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075'
GT = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/'
      'teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')

pipe = Trellis2TexturingPipeline.from_pretrained(
    'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json'); pipe.cuda()
mesh = trimesh.load(f'{R}/frozen_f0075.ply', process=False, force='mesh')
mpp = pipe.preprocess_mesh(mesh)
v_raw = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
v_pp = torch.from_numpy(np.asarray(mpp.vertices)).float().cuda()
faces = torch.from_numpy(np.asarray(mesh.faces)).int().cuda().contiguous()

torch.manual_seed(42)
cond = pipe.get_cond([pipe.preprocess_image(Image.open(GT).convert('RGB'))], RES)
ss = pipe.encode_shape_slat(mpp, RES)
slat = pipe.sample_tex_slat(cond, pipe.models[f'tex_slat_flow_model_{RES}'], ss, {})
pbr = pipe.decode_tex_slat(slat)
F = pbr.feats.float()
shape = torch.Size([*pbr.shape, *pbr.spatial_shape])
print(f'\n[FIELD] pbr.shape={tuple(pbr.shape)}  spatial={tuple(pbr.spatial_shape)}  '
      f'feats={tuple(F.shape)}')
print(f'[FIELD] stored base_color mean={F[:,:3].mean():.4f}  '
      f'p99={F[:,:3].flatten().quantile(0.99):.4f}')

def samp(g):
    return grid_sample_3d(pbr.feats, pbr.coords, shape=shape,
                          grid=g.reshape(1, -1, 3), mode='trilinear')[..., :3].float()

# ---- ground truth: round trip, compared PER VOXEL, not in aggregate ---------
sel = torch.randperm(F.shape[0], device='cuda')[:20000]
gt_feats = F[sel][..., :3]
rt = samp(pbr.coords[sel, 1:].float()).reshape(-1, 3)
err = (rt - gt_feats).abs()
print(f'\n[ROUND TRIP] per-voxel |sampled - stored|: mean={err.mean():.5f} '
      f'max={err.max():.5f}   corr={torch.corrcoef(torch.stack([rt.flatten(), gt_feats.flatten()]))[0,1]:.4f}')
print(f'             stored subset mean={gt_feats.mean():.4f}  sampled mean={rt.mean():.4f}')

# ---- the renderer's exact geometry -----------------------------------------
_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXT = torch.tensor([[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,1.,0.,2.],[0.,0.,0.,1.]]).cuda()
P = torch.zeros((4,4)).cuda()
P[0,0]=2*_FX; P[1,1]=2*_FX; P[0,2]=0.; P[1,2]=0.
P[2,2]=3.0/2.5; P[2,3]=0.5*3.0/(0.5-3.0); P[3,2]=1.
full = (P @ EXT).unsqueeze(0)
vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
clip = torch.bmm(vh, full.transpose(-1,-2)).contiguous()
rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, faces, (IMG, IMG))
m = rast[0, ..., 3] > 0
midx = torch.nonzero(m.reshape(-1)).squeeze(1)
pos = dr.interpolate(v_pp.unsqueeze(0).contiguous(), rast, faces)[0][0].reshape(-1,3)[midx]

for name, pts in [('all vertices', v_pp),
                  ('interpolated pixels (renderer)', pos)]:
    o = samp((pts + 0.5) * RES)
    print(f'\n[{name}] n={pts.shape[0]}')
    print(f'  sampled mean={o.mean():.4f} p99={o.flatten().quantile(0.99):.4f} '
          f'frac>0.05={float((o.max(1).values>0.05).float().mean()):.3f}')
    g = (pts + 0.5) * RES
    print(f'  grid min={g.min(0).values.tolist()}')
    print(f'  grid max={g.max(0).values.tolist()}')


# ---- the payoff: composite and save what the renderer actually produces -----
import torchvision.utils as _tvu
for FR in [1, 5, 40, 75, 120]:
    gt = Image.open(GT.replace('frame_0075', f'frame_{FR:04d}')).convert('RGB')
    a = np.asarray(gt).astype(np.float32) / 255.0
    al = (a.min(axis=2) < 245/255).astype(np.float32)
    ci = Image.fromarray(((a * al[..., None]) * 255).astype(np.uint8))
    torch.manual_seed(42)
    c = pipe.get_cond([pipe.preprocess_image(ci)], RES)
    sl = pipe.sample_tex_slat(c, pipe.models[f'tex_slat_flow_model_{RES}'], ss, {})
    pv = pipe.decode_tex_slat(sl)
    sh = torch.Size([*pv.shape, *pv.spatial_shape])
    at = grid_sample_3d(pv.feats, pv.coords, shape=sh,
                        grid=((pos + 0.5) * RES).reshape(1, -1, 3), mode='trilinear')
    rgb = at[..., :3].reshape(-1, 3).float().clamp(0, 1)
    base = torch.ones(IMG * IMG, 3, device='cuda')
    im = base.index_put((midx,), rgb).view(IMG, IMG, 3)
    g = torch.from_numpy(np.array(gt.resize((IMG, IMG), Image.LANCZOS))).float().cuda() / 255
    strip = torch.cat([g, im], dim=1).permute(2, 0, 1)
    _tvu.save_image(strip, f'/net/projects/ranalab/rajhansini/TRELLIS/experiments/'
                           f'trellis2_baseline/out/render_check_f{FR:04d}.png')
    print(f'[RENDER f{FR:04d}] sampled mean={rgb.mean():.4f} '
          f'p99={rgb.flatten().quantile(0.99):.4f} max={rgb.max():.4f}', flush=True)
