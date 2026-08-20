"""
unwrap_uv.py — give the odedstein meshes UVs with xatlas.

WHY
  None of hand / mushroom / skull / sword / wingnut ship UVs. TRELLIS.2's
  postprocess_mesh falls back to cumesh.uv_unwrap when mesh.visual.uv is absent,
  and that raises CUDA 209 on this cluster. Every mesh already in use here
  (spot_render_frame.obj, the multi_iSeg *_unwarp.glb) carries UVs for that reason.

WHAT IT CHANGES
  xatlas splits vertices along seams, so the OUTPUT HAS MORE VERTICES than the
  input. Faces index the new vertex array and the surface is identical — but any
  code that assumes v_raw[i] and v_uv[i] are the same vertex will be wrong.
  frozen_f0075_uv.obj went 215,462 -> 262,584 the same way.

  Originals are never overwritten; output is <name>_uv.obj alongside them.
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import xatlas

HERE = Path(__file__).resolve().parent
targets = sys.argv[1:] or ['hand', 'mushroom', 'skull', 'sword', 'wingnut']

print(f'{"mesh":<26} {"verts in":>10} {"verts out":>10} {"faces":>9} {"charts":>7}  status')
for name in targets:
    d = HERE / name
    for src in sorted(d.glob('*.obj')):
        if src.stem.endswith('_uv'):
            continue
        dst = src.with_name(src.stem + '_uv.obj')
        try:
            m = trimesh.load(src, process=False, force='mesh')
            v = np.asarray(m.vertices, dtype=np.float32)
            f = np.asarray(m.faces, dtype=np.uint32)

            atlas = xatlas.Atlas()
            atlas.add_mesh(v, f)
            atlas.generate()
            vmap, indices, uvs = atlas[0]          # vmap: new->old vertex index

            out = trimesh.Trimesh(
                vertices=v[vmap],
                faces=indices.astype(np.int64),
                visual=trimesh.visual.TextureVisuals(uv=uvs),
                process=False,
            )
            out.export(dst)

            chk = trimesh.load(dst, process=False, force='mesh')
            has_uv = getattr(getattr(chk, 'visual', None), 'uv', None) is not None
            same_faces = len(chk.faces) == len(m.faces)
            ok = 'OK' if (has_uv and same_faces) else \
                 f'CHECK uv={has_uv} faces_match={same_faces}'
            print(f'{src.parent.name + "/" + src.name:<26} {len(v):>10,} '
                  f'{len(out.vertices):>10,} {len(out.faces):>9,} '
                  f'{atlas.chart_count:>7,}  {ok}')
        except Exception as e:
            print(f'{src.parent.name + "/" + src.name:<26} {"":>10} {"":>10} {"":>9} '
                  f'{"":>7}  FAILED: {type(e).__name__}: {e}')
