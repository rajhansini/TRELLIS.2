"""
diag_pos.py — are the per-pixel interpolated surface positions actually correct?

diag_sampling.py proved the coordinate convention is right: sampling the PBR
field at mesh VERTICES with (v+0.5)*RES gives mean 0.4037, matching a round-trip
at the field's own voxel coords (0.4063), with median distance 0.54 voxels.

But SurfaceRenderer samples at PER-PIXEL interpolated positions and gets 0.0364,
with p99 0.0396 and max 0.1080 — a nearly CONSTANT value. Constant is the tell:
it is not interpolation dragging values toward zero, it is every pixel sampling
the same place. If dr.interpolate returned ~0, every pixel would map to
(0+0.5)*512 = 256, the object's centre, and produce exactly this.

So: compare the statistics of the interpolated positions against the vertex
positions they are interpolated from.
"""
import os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
import numpy as np, torch, trimesh, math
import nvdiffrast.torch as dr
from trellis2.pipelines import Trellis2TexturingPipeline

R = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075'
RES_IMG = 512
pipe = Trellis2TexturingPipeline.from_pretrained(
    'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
mesh = trimesh.load(f'{R}/frozen_f0075.ply', process=False, force='mesh')
mesh_pp = pipe.preprocess_mesh(mesh)
v_raw = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().cuda()
faces = torch.from_numpy(np.asarray(mesh.faces)).int().cuda().contiguous()
print(f'v_raw bounds min={v_raw.min(0).values.tolist()} max={v_raw.max(0).values.tolist()}')
print(f'v_pp  bounds min={v_pp.min(0).values.tolist()} max={v_pp.max(0).values.tolist()}')

_FX = 1.0 / (2.0 * math.tan(math.radians(40.0 / 2)))
EXT = torch.tensor([[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,1.,0.,2.],[0.,0.,0.,1.]]).cuda()
INT = torch.tensor([[_FX,0.,0.5],[0.,_FX,0.5],[0.,0.,1.]]).cuda()
near, far = 0.5, 3.0
P = torch.zeros((4,4)).cuda()
P[0,0]=2*INT[0,0]; P[1,1]=2*INT[1,1]; P[0,2]=2*INT[0,2]-1; P[1,2]=-2*INT[1,2]+1
P[2,2]=far/(far-near); P[2,3]=near*far/(near-far); P[3,2]=1.
full = (P @ EXT).unsqueeze(0)

vh = torch.cat([v_raw, torch.ones_like(v_raw[:, :1])], -1).unsqueeze(0)
clip = torch.bmm(vh, full.transpose(-1,-2)).contiguous()
ctx = dr.RasterizeCudaContext()
rast, _ = dr.rasterize(ctx, clip, faces, (RES_IMG, RES_IMG))
mask = rast[0, ..., 3] > 0
midx = torch.nonzero(mask.reshape(-1)).squeeze(1)
print(f'\nrasterised: {int(mask.sum())} px covered ({100*mask.float().mean():.1f}%)')

for tag, attr in [('v_pp  (what we interpolate)', v_pp),
                  ('v_raw (sanity check)',        v_raw)]:
    out = dr.interpolate(attr.unsqueeze(0).contiguous(), rast, faces)[0]
    p = out[0].reshape(-1, 3)[midx]
    print(f'\n{tag}')
    print(f'  interpolated min={p.min(0).values.tolist()}')
    print(f'  interpolated max={p.max(0).values.tolist()}')
    print(f'  mean={p.mean(0).tolist()}  std={p.std(0).tolist()}')
    print(f'  frac exactly zero rows: {float((p.abs().sum(1)==0).float().mean()):.4f}')
