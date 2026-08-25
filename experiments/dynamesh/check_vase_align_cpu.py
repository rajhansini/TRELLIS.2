"""check_vase_align_cpu.py -- gate the vase pose BEFORE the 4-GPU-hour job runs.

Same camera as the pipeline (verify_known_rotations.py): fov 40, the confirmed
EXTRINSICS with cam_y=(0,0,-1) and t=2, res 960. Same canonical() axis map. Same
video_mask() rule (border median + largest connected component). CPU rasteriser,
silhouette only -- no nvdiffrast, no GPU, runs in seconds.

COVERAGE is the gate, not symmetric IoU: a cast shadow only ever ADDS to the video
mask, so IoU punishes it while coverage ("is any of OUR silhouette outside the
video's object") cannot be fooled by one. That is the lesson from
teapot_ceramic_crack, whose search chased its own shadow.
"""
import json, math
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
OBJ, RES, TOL = 'vase_floral', 960, 12
_FX = 1.0/(2.0*math.tan(math.radians(40.0/2)))
EXT = np.array([[1., 0, 0, 0], [0, 0, -1., 0], [0, 1., 0, 2.], [0, 0, 0, 1.]])


def canonical(v):
    lo, hi = v.min(0), v.max(0)
    v = (v - (lo+hi)/2) * (0.99999/(hi-lo).max())
    return v[:, (0, 2, 1)] * np.array([-1., 1., 1.])


def silhouette(V, F, res=RES):
    """Delegates to the CALIBRATED rasteriser.

    The first version of this function projected with pixel_y = -ndc_y, guessing
    nvdiffrast's image-y direction. It guessed wrong: calib_cpu_raster.py scored
    both conventions against the render_mask.npy nvdiffrast itself wrote and
    flip_y=False matches at IoU 0.9943 (spot_lava) / 0.9917 (teapot_porcelain)
    while flip_y=True gives 0.77 / 0.51. The wrong sign made every known-good
    object look 0.2-0.5 IoU worse than recorded and produced a FALSE FAIL on the
    vase. One implementation now, calibrated, imported rather than duplicated.
    """
    from calib_cpu_raster import raster
    return raster(np.asarray(V, float), np.asarray(F), res, flip_y=False)


def largest(m):
    lb, n = ndimage.label(m)
    return m if n <= 1 else lb == (1+int(np.argmax(np.bincount(lb.ravel())[1:])))


def video_mask(p, res=RES, tol=TOL):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    m = largest(np.abs(a-bg).max(2) > tol)
    if m.shape[0] != res:
        m = np.asarray(Image.fromarray((m*255).astype(np.uint8)).resize((res, res), Image.NEAREST)) > 127
    return m, bg


R = np.array(json.load(open(E/'out/orient/orientation_vase.json'))[0]['R'])
mesh = trimesh.load(T2/f'data/{OBJ}/mesh/vase.obj', process=False, force='mesh')
V = canonical(np.asarray(mesh.vertices, float).copy()) @ R.T
F = np.asarray(mesh.faces)
ours = silhouette(V, F)

rows = []
for fi in (1, 75, 150):
    p = T2/f'data/{OBJ}/frames_from_video/frame_{fi:04d}.png'
    vid, bg = video_mask(p)
    inter = (ours & vid).sum(); union = (ours | vid).sum()
    outside = (ours & ~vid).sum()
    rows.append((fi, ours.sum(), vid.sum(), inter/union, outside/ours.sum(), bg))
    print(f"frame {fi:4d}  ours={ours.sum():7d}px  video={vid.sum():7d}px  "
          f"IoU={inter/union:.4f}  OUTSIDE={100*outside/ours.sum():5.2f}%  bg={bg}")

worst_out = max(r[4] for r in rows); best_iou = max(r[3] for r in rows)
Image.fromarray(np.dstack([(ours*255).astype(np.uint8),
                           (video_mask(T2/f'data/{OBJ}/frames_from_video/frame_0001.png')[0]*255).astype(np.uint8),
                           np.zeros((RES, RES), np.uint8)])).save(E/'out/vase_align_overlay.png')
print(f"\nworst outside-coverage {100*worst_out:.2f}%   best IoU {best_iou:.4f}")
print(f"overlay (red=ours, green=video, yellow=agree) -> {E/'out/vase_align_overlay.png'}")
print("VERDICT:", "PASS" if worst_out < 0.05 else "FAIL — pose is wrong, cancel the chain")
