"""
export_rung_glb.py — write a trained arm's textured mesh out as GLB/PLY

WHY VERTEX COLOURS AND NOT A UV BAKE
  run_baseline.py bakes into a 2048^2 atlas after an xatlas unwrap. This mesh has
  430,910 faces, which is 9.7 texels per face — the bake aliases and speckles,
  and it is what made an earlier TRELLIS.2 export look worse than TRELLIS v1
  despite the field being fine. The PBR field is sampled DIRECTLY at the mesh's
  215,462 vertices instead. That is 7x the 29,349 pixels the training camera
  actually resolves, so vertex colour is not the limiting factor here — the
  atlas was.

TWO FRAMES, ONE VERTEX ORDER (same convention as the trainer)
  v_raw  the registered mesh as exported; the GLB carries THESE positions.
  v_pp   preprocess_mesh output, normalised to [-0.5, 0.5]; the PBR voxel field
         lives in this frame, so sampling happens here.
  preprocess_mesh preserves vertex order, which is what lets the two coexist.

Writes both the frozen and the adapted mesh from the SAME noise and the SAME
conditioning, so anything that differs between them is the adapter.
"""

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['HF_HOME'] = '/net/scratch/rajhansini/.cache/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '/net/projects/ranalab/rajhansini/TRELLIS.2')
_HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='a runs/rungNN_... directory')
ap.add_argument('--ckpt', default='lora_best.pt')
ap.add_argument('--frames', default='1',
                help="frame indices, separated by ':' or ','. Prefer ':' when "
                     "passing through sbatch --export, which itself splits on "
                     "commas and would silently truncate the list to the first "
                     "entry.")
ap.add_argument('--tag', default=None)
ARGS = ap.parse_args()

RUN_DIR = Path(ARGS.run)
CFG = json.load(open(RUN_DIR / 'config.json'))
FRAMES = [int(x) for x in ARGS.frames.replace(':', ',').split(',') if x.strip()]

_so, _se = sys.stdout, sys.stderr
sys.argv = ['rung17_backproj_lora.py',
            '--mesh', MESH,
            '--rank', str(CFG['rank']), '--targets', CFG['targets'],
            '--seed', str(CFG['seed']), '--epochs', str(CFG['epochs']),
            '--n-frames', str(CFG['n_frames']), '--resolution', str(CFG['resolution'])]
sys.path.insert(0, str(_HERE))
import rung17_backproj_lora as R          # noqa: E402
sys.stdout, sys.stderr = _so, _se

import numpy as np
import torch
import trimesh
from PIL import Image

DEVICE = torch.device('cuda')
RUNG = CFG.get('rung', '?')
TGT = '+'.join(t.replace('to_', '') for t in CFG.get('target_set', []))
OUT = (_HERE / 'out' / (ARGS.tag or f'rung{RUNG}_glb')).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def main():
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis2.modules.sparse.conv import config as conv_config
    from flex_gemm.ops.grid_sample import grid_sample_3d
    conv_config.FLEX_GEMM_ALGO = 'implicit_gemm_splitk'

    log('=' * 88)
    log(f'EXPORT GLB — rung{RUNG} ({TGT})   frames {FRAMES}')
    log('=' * 88)

    pipe = Trellis2TexturingPipeline.from_pretrained(
        'microsoft/TRELLIS.2-4B', config_file='texturing_pipeline.json')
    pipe.low_vram = False
    pipe.cuda()
    flow = pipe.models[f'tex_slat_flow_model_{CFG["resolution"]}']
    dec = pipe.models['tex_slat_decoder']
    for m in pipe.models.values():
        if isinstance(m, torch.nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)

    mesh_in = trimesh.load('/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/'
                           'frozen_f0075.ply', process=False, force='mesh')
    mesh_pp = pipe.preprocess_mesh(mesh_in)
    assert len(mesh_pp.vertices) == len(mesh_in.vertices), 'vertex order broken'
    v_pp = torch.from_numpy(np.asarray(mesh_pp.vertices)).float().to(DEVICE)
    V = len(mesh_in.vertices)
    log(f'[MESH] {V:,} verts  {len(mesh_in.faces):,} faces')

    reg = R.LoRARegistry(len(flow.blocks), flow.model_channels, flow.cond_channels,
                         CFG['rank'], with_mlp=False,
                         mlp_hidden=int(flow.model_channels * flow.mlp_ratio),
                         targets=tuple(CFG['target_set']),
                         active=CFG.get('active')).to(DEVICE)
    st = torch.load(RUN_DIR / 'ckpts' / ARGS.ckpt, map_location=DEVICE, weights_only=False)
    reg.load_state_dict(st['reg'] if 'reg' in st else st['registry_state'])
    reg.eval()
    log(f'[LORA] {sum(p.numel() for p in reg.parameters()):,} params  '
        f'epoch={st.get("epoch")}')

    shape_slat = pipe.encode_shape_slat(mesh_pp, CFG['resolution'])
    ss_std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(DEVICE)
    ss_mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(DEVICE)
    ss_n = (shape_slat - ss_mean) / ss_std
    tex_std = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(DEVICE)
    tex_mean = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(DEVICE)

    gt_dir = Path(R.args.gt_dir)
    allf = list(range(1, CFG['n_frames'] + 1))
    raws = [np.array(Image.open(gt_dir / f'frame_{f:04d}.png').convert('RGB')) for f in allf]
    als = [R.largest_component(r.min(axis=2) < 245) for r in raws]
    u = np.zeros_like(als[0])
    for a in als:
        u |= a
    ys, xs = np.where(u)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    bbox = (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))

    def cond_image(fi):
        i = fi - 1
        rgba = np.concatenate([raws[i], (als[i] * 255).astype(np.uint8)[..., None]], -1)
        f = np.asarray(Image.fromarray(rgba).crop(bbox)).astype(np.float32) / 255.0
        return Image.fromarray(((f[:, :, :3] * f[:, :, 3:4]) * 255).astype(np.uint8))

    g = torch.Generator(device='cpu').manual_seed(CFG['seed'])
    noise = ss_n.replace(feats=torch.randn(
        ss_n.coords.shape[0], flow.in_channels - ss_n.feats.shape[1],
        generator=g).to(DEVICE))
    grid_res = CFG['resolution']

    def vert_colours(pbr):
        attrs = grid_sample_3d(pbr.feats, pbr.coords,
                               shape=torch.Size([*pbr.shape, *pbr.spatial_shape]),
                               grid=((v_pp + 0.5) * grid_res).reshape(1, -1, 3),
                               mode='trilinear')
        return attrs[..., :3].reshape(-1, 3).float().clamp(0, 1).cpu().numpy()

    def write(cols, path):
        m = trimesh.Trimesh(vertices=np.asarray(mesh_in.vertices),
                            faces=np.asarray(mesh_in.faces),
                            vertex_colors=(cols * 255).astype(np.uint8),
                            process=False)
        m.export(path)
        return path

    for fi in FRAMES:
        with torch.no_grad():
            c = pipe.get_cond([cond_image(fi)], CFG['resolution'])['cond']
            pbr_fz = dec(R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
                         * tex_std + tex_mean) * 0.5 + 0.5
            with R.lora_ctx(flow, reg):
                xl = R.run_ode(flow, noise, c, ss_n, R.T_PAIRS)
            pbr_lo = dec(xl * tex_std + tex_mean) * 0.5 + 0.5
        cf, cl = vert_colours(pbr_fz), vert_colours(pbr_lo)
        log(f'\n[FRAME {fi:04d}]  frozen mean {cf.mean():.4f}  '
            f'rung{RUNG} mean {cl.mean():.4f}')
        log(f'  vertices >0.95 on all 3 channels:  frozen '
            f'{int((cf.min(1) > 0.95).sum()):,}   rung{RUNG} '
            f'{int((cl.min(1) > 0.95).sum()):,}  of {V:,}')
        for ext in ('glb', 'ply'):
            log(f'  {write(cf, OUT / f"frozen_f{fi:04d}.{ext}")}')
            log(f'  {write(cl, OUT / f"rung{RUNG}_{TGT}_f{fi:04d}.{ext}")}')
    log(f'\n[SAVE] {OUT}\n[DONE]')


if __name__ == '__main__':
    main()
