"""Re-render the conditioning images in the PIPELINE camera (fov 40, dist 2).

WHY. front/ was rendered with the Kling camera copied from render_final.py --
fov 30, dist 2.6. But solve_orientation.py and the trainer both use fov 40, dist 2
(EXTRINSICS z=2, _FX_N = 1/(2 tan 20 deg)). The alignment solver searches ROTATION
only, on top of canonical centring and uniform scale, so it cannot absorb a field-of-
view difference: batch D landed at IoU 0.89-0.93 where spot_lava -- which was rendered
in the pipeline camera to begin with -- sits at 0.9913, and gargoyle fell through the
0.80 gate entirely at 0.746 (silhouette 0.789x the video's width but 0.978x its height,
non-uniform, which is the signature of an fov mismatch and not of scale or translation).

Same meshes, same chosen view angles as front_views.tsv -- only the camera changes.
"""
import csv, pathlib, sys
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
g = {'__file__': str(HERE / 'best_view.py'), '__name__': 'lib'}
src = open(HERE / 'best_view.py').read().split('YAWS = range')[0]
src = src.replace('RES, DIST, FOV = 1024, 2.6, 30.0',
                  'RES, DIST, FOV = 1024, 2.0, 40.0')      # <- the pipeline camera
exec(src, g)
load_obj, normalize, sample, render, UPS = (g['load_obj'], g['normalize'],
                                            g['sample'], g['render'], g['UPS'])
assert g['DIST'] == 2.0 and g['FOV'] == 40.0, 'camera override did not take'

OUT = HERE / 'front_pipecam'; OUT.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(HERE / 'front_views.tsv'), delimiter='\t'))
want = set(sys.argv[1:]) or None
for r in rows:
    if want and r['mesh'] not in want:
        continue
    V, F = load_obj(HERE / 'OBJ' / f"{r['mesh']}.obj")
    V = normalize(V)
    P, N = sample(V @ UPS[r['up']].T, F, 1_500_000, seed=1)
    Image.fromarray(render(P, N, int(r['yaw']), int(r['pitch']))).save(OUT / f"{r['mesh']}.png")
    print(f"{r['mesh']:10s} up={r['up']:2s} yaw={r['yaw']:>3s} pitch={r['pitch']:>2s} "
          f"-> front_pipecam/{r['mesh']}.png  (fov 40, dist 2)", flush=True)
