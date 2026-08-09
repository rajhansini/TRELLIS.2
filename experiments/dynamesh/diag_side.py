"""
diag_side.py — is the generated texture on the side our camera CANNOT see?

diag_pos.py showed the interpolated surface positions are correct and span only
the camera-facing half: v_pp z in [-0.310, +0.040] out of a full [-0.310, +0.310].
Sampling the field at ALL vertices gives 0.4037; sampling only that visible half
gives 0.0364. If the texture were uniformly distributed those would agree.

So: split the vertices by the sign of v_pp z and compare. If +z is bright and -z
is dark, TRELLIS.2 textured the far side and our v1 camera is looking at the
back of the asset — a frame-convention mismatch, not a texture failure.
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
v = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().cuda()

torch.manual_seed(42)
cond = pipe.get_cond([pipe.preprocess_image(Image.open(GT).convert('RGB'))], RES)
ss = pipe.encode_shape_slat(mesh_pp, RES)
slat = pipe.sample_tex_slat(cond, pipe.models[f'tex_slat_flow_model_{RES}'], ss, {})
pbr = pipe.decode_tex_slat(slat)
shape = torch.Size([*pbr.shape, *pbr.spatial_shape])

def s(sel, name):
    g = ((v[sel] + 0.5) * RES).reshape(1, -1, 3)
    o = grid_sample_3d(pbr.feats, pbr.coords, shape=shape, grid=g, mode='trilinear')[..., :3].float()
    print(f'  {name:34} n={int(sel.sum()):>7}  mean={o.mean():.4f}  '
          f'p99={o.flatten().quantile(0.99):.4f}  frac>0.05={float((o.max(dim=1).values>0.05).float().mean()):.3f}')

print('\n[SPLIT BY AXIS — which half of the asset carries the texture?]')
for ax, nm in [(0, 'x'), (1, 'y'), (2, 'z')]:
    print(f' axis {nm}:')
    s(v[:, ax] < 0, f'  {nm} < 0')
    s(v[:, ax] > 0, f'  {nm} > 0')
print('\n our camera sees the z<0 half (measured in diag_pos.py)')
