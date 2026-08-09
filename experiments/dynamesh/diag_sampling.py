"""
diag_sampling.py — where are our surface sample points relative to the field?

GATE-sample measured: PBR field base_color mean 0.1201 / p99 0.3835, but our
surface samples mean 0.0364 with frac>0.05 == 0.000. grid_sample_3d returns 0
outside occupied voxels, so samples that are uniformly dim — not zero, dim —
mean we are landing BETWEEN occupied voxels and empty space and trilinear
interpolation is dragging everything toward zero.

This measures the geometry directly:
  1. Sample the field AT its own voxel coordinates. That must round-trip to the
     stored features; if it does not, our grid convention is wrong, full stop.
  2. Compute, for our surface points, the distance in voxel units to the nearest
     occupied voxel. Near 0 means the coordinates are right and something else
     is wrong; large means the frame/scale/axis-order is wrong.
  3. Try the obvious alternative conventions and report which one puts the
     surface points on the voxels.
"""
import os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
import numpy as np, torch, trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline
from flex_gemm.ops.grid_sample import grid_sample_3d

RES = 512
R = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075'
GT = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/'
      'teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')

pipe = Trellis2TexturingPipeline.from_pretrained(
    'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
pipe.cuda()
mesh = trimesh.load(f'{R}/frozen_f0075.ply', process=False, force='mesh')
mesh_pp = pipe.preprocess_mesh(mesh)
v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().cuda()
print(f'[MESH] preprocessed bounds min={v_pp.min(0).values.tolist()} '
      f'max={v_pp.max(0).values.tolist()}', flush=True)

img = Image.open(GT).convert('RGB')
torch.manual_seed(42)
cond = pipe.get_cond([pipe.preprocess_image(img)], RES)
ss = pipe.encode_shape_slat(mesh_pp, RES)
slat = pipe.sample_tex_slat(cond, pipe.models[f'tex_slat_flow_model_{RES}'], ss, {})
pbr = pipe.decode_tex_slat(slat)
C = pbr.coords            # [N, 4] batch + xyz voxel indices
F = pbr.feats.float()
print(f'\n[FIELD] voxels={tuple(F.shape)} spatial_shape={tuple(pbr.spatial_shape)}')
print(f'[FIELD] coord ranges  x[{C[:,1].min()},{C[:,1].max()}]  '
      f'y[{C[:,2].min()},{C[:,2].max()}]  z[{C[:,3].min()},{C[:,3].max()}]')
print(f'[FIELD] base_color mean={F[:,:3].mean():.4f} p99='
      f'{F[:,:3].flatten().quantile(0.99):.4f}')

shape = torch.Size([*pbr.shape, *pbr.spatial_shape])

def probe(name, grid):
    out = grid_sample_3d(pbr.feats, pbr.coords, shape=shape,
                         grid=grid.reshape(1, -1, 3), mode='trilinear')[..., :3].float()
    print(f'  {name:38} mean={out.mean():.4f}  p99={out.flatten().quantile(0.99):.4f}  '
          f'frac>0.05={float((out.max(dim=1).values > 0.05).float().mean()):.3f}')

print('\n[1] ROUND TRIP: sample at the field\'s OWN voxel coords (must match field)')
probe('at own coords (+0.5 centre)', C[:, 1:].float() + 0.5)
probe('at own coords (no offset)',   C[:, 1:].float())

print('\n[2] OUR CONVENTION and alternatives, at mesh VERTICES')
probe('(v+0.5)*RES            [ours]', (v_pp + 0.5) * RES)
probe('(v+0.5)*RES, zyx order',       ((v_pp + 0.5) * RES).flip(-1))
sp = torch.tensor(list(pbr.spatial_shape), dtype=torch.float32, device='cuda')
probe('(v+0.5)*spatial_shape',        (v_pp + 0.5) * sp)
probe('(v+0.5)*spatial_shape, zyx',   ((v_pp + 0.5) * sp).flip(-1))

print('\n[3] DISTANCE from our sample points to the nearest occupied voxel')
ours = (v_pp + 0.5) * RES
sub = ours[torch.randperm(ours.shape[0], device='cuda')[:2000]]
cf = C[:, 1:].float()
d = torch.cdist(sub, cf).min(dim=1).values
print(f'  ours   : median={d.median():.2f} vox  mean={d.mean():.2f}  max={d.max():.2f}')
print(f'  centre of our points   {ours.mean(0).tolist()}')
print(f'  centre of the voxels   {cf.mean(0).tolist()}')
print(f'  our  bbox  min={ours.min(0).values.tolist()} max={ours.max(0).values.tolist()}')
print(f'  vox  bbox  min={cf.min(0).values.tolist()} max={cf.max(0).values.tolist()}')
