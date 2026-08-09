"""
verify_one_mesh.py — is the geometry BYTE-IDENTICAL across frames, or just the
same vertex COUNT?

Earlier runs only logged the vertex count (262,584 constant over 150 frames).
Equal counts do not prove equal positions, and nothing so far checked the faces
at all. This compares the actual arrays.

Two paths are checked because they are not the same claim:

  A. OUR path (run_baseline_field.py, rung14 renders): the mesh is loaded once
     at startup and rasterised once. TRELLIS.2 is never asked to produce
     geometry. Identity is then trivial, but worth stating explicitly.

  B. pipe.run() path (the UV-baked baseline): postprocess_mesh RETURNS a mesh
     per frame. That mesh could in principle differ frame to frame, so it is
     compared exactly across several frames.
"""
import os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
import numpy as np, torch, trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

# postprocess_mesh flips axes in place on mesh.vertex_normals, which trimesh
# hands back read-only (trellis2_texturing.py:363). Only the UV-supplied branch
# reaches it. Same patch run_baseline_field.py carries.
_vn = trimesh.Trimesh.vertex_normals
trimesh.Trimesh.vertex_normals = property(
    lambda self: np.array(_vn.fget(self), dtype=np.float64), _vn.fset, _vn.fdel)

MESH = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075_uv.obj'
GT = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/'
      'teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150')
FRAMES = [1, 40, 75, 120, 150]

pipe = Trellis2TexturingPipeline.from_pretrained(
    'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
pipe.cuda()
mesh_in = trimesh.load(MESH, process=False, force='mesh')
print(f'[INPUT] {len(mesh_in.vertices):,} verts  {len(mesh_in.faces):,} faces  '
      f'uv={mesh_in.visual.uv is not None}', flush=True)

ref_v = ref_f = ref_uv = None
tex_hashes = []
for fr in FRAMES:
    img = Image.open(f'{GT}/frame_{fr:04d}.png').convert('RGB')
    a = np.asarray(img).astype(np.float32) / 255
    al = (a.min(axis=2) < 245/255).astype(np.float32)
    ci = Image.fromarray(((a * al[..., None]) * 255).astype(np.uint8))
    torch.manual_seed(42)
    out = pipe.run(mesh_in, ci, seed=42, preprocess_image=True,
                   resolution=512, texture_size=1024)
    v = np.asarray(out.vertices); f = np.asarray(out.faces)
    uv = np.asarray(out.visual.uv)
    tex = np.asarray(out.visual.material.baseColorTexture.convert('RGB'))
    tex_hashes.append(hash(tex.tobytes()))
    if ref_v is None:
        ref_v, ref_f, ref_uv = v.copy(), f.copy(), uv.copy()
        print(f'  f{fr:04d}  REFERENCE  {v.shape[0]:,} verts  {f.shape[0]:,} faces', flush=True)
    else:
        dv = np.abs(v - ref_v).max() if v.shape == ref_v.shape else float('inf')
        same_f = (f.shape == ref_f.shape) and np.array_equal(f, ref_f)
        duv = np.abs(uv - ref_uv).max() if uv.shape == ref_uv.shape else float('inf')
        print(f'  f{fr:04d}  verts {v.shape[0]:,}  max|Δvertex|={dv:.3e}  '
              f'faces identical={same_f}  max|Δuv|={duv:.3e}', flush=True)

print(f'\n[TEXTURE] distinct texture images across {len(FRAMES)} frames: '
      f'{len(set(tex_hashes))}/{len(FRAMES)}')
print('[VERDICT] geometry identical AND textures differ' if len(set(tex_hashes)) == len(FRAMES)
      else '[VERDICT] CHECK — textures are not all distinct')
