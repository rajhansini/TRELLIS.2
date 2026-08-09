"""
diag_black_texture.py — why is the generated texture black?

The baseline's own texture map (via TRELLIS.2's postprocess_mesh, not our renderer)
came out near-black even though the conditioning image is a clearly visible warm
teapot (frame 75 foreground mean RGB [113.7, 50.8, 26.7]).

grid_sample_3d returns exactly 0 outside the occupied sparse voxels, so "black"
is the signature of sampling where the field does not exist, rather than of a
genuinely dark texture. Two suspects, crossed here:

  MESH   the UV-unwrapped .obj splits vertices along seams (215,462 -> 262,584).
         encode_shape_slat runs o_voxel.mesh_to_flexible_dual_grid on it, and a
         seam-split mesh may voxelize differently from the original .ply.
  COND   our deterministic alpha+crop vs the pipeline's own BiRefNet preprocess.

Reports, per variant: the shape latent, the decoded PBR field's own statistics
(before any sampling), and the baked texture's. If the FIELD is bright but the
TEXTURE is dark, it is a sampling/coordinate fault; if the field itself is dark,
it is generation.
"""
import os, sys, json
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')

import numpy as np, torch, trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

R = '/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075'
GT = ('/net/projects/ranalab/rajhansini/MV-Adapter-Experimental/outputs/'
      'teapot_lava_kling_premium/teapot_lava_kling_premium_front/all_frames_150/frame_0075.png')

pipe = Trellis2TexturingPipeline.from_pretrained(
    'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
pipe.cuda()
raw = Image.open(GT).convert('RGB')

for mesh_tag, mesh_path in [('orig_ply', f'{R}/frozen_f0075.ply'),
                            ('uv_obj',   f'{R}/frozen_f0075_uv.obj')]:
    for cond_tag in ['pipeline_rembg', 'ours_deterministic']:
        try:
            mesh = trimesh.load(mesh_path, process=False, force='mesh')
            if cond_tag == 'pipeline_rembg':
                img, pre = raw, True
            else:
                a = np.asarray(raw).astype(np.float32) / 255.0
                al = (a.min(axis=2) < 245/255).astype(np.float32)
                img = Image.fromarray(((a * al[..., None]) * 255).astype(np.uint8))
                pre = False

            mesh_pp = pipe.preprocess_mesh(mesh)
            im = pipe.preprocess_image(img) if pre else img
            torch.manual_seed(42)
            cond = pipe.get_cond([im], 512)
            ss = pipe.encode_shape_slat(mesh_pp, 512)
            slat = pipe.sample_tex_slat(cond, pipe.models['tex_slat_flow_model_512'], ss, {})
            pbr = pipe.decode_tex_slat(slat)
            f = pbr.feats.float()
            print(f'\n=== mesh={mesh_tag}  cond={cond_tag} ===', flush=True)
            print(f'  shape_slat  {tuple(ss.feats.shape)}   pbr voxels {tuple(f.shape)}')
            print(f'  spatial_shape {tuple(pbr.spatial_shape)}')
            print(f'  PBR FIELD base_color: mean={f[:,:3].mean():.4f} '
                  f'max={f[:,:3].max():.4f} p99={f[:,:3].flatten().quantile(0.99):.4f}')
            print(f'  PBR FIELD alpha     : mean={f[:,5].mean():.4f} max={f[:,5].max():.4f}')
            out = pipe.postprocess_mesh(mesh_pp, pbr, 512, 1024)
            t = np.asarray(out.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)
            print(f'  BAKED TEXTURE       : mean={t.mean():.1f} max={t.max():.0f} '
                  f'frac>32: {(t.max(axis=2) > 32).mean():.3f}')
        except Exception as e:
            print(f'\n=== mesh={mesh_tag}  cond={cond_tag} === FAILED {type(e).__name__}: {e}',
                  flush=True)
