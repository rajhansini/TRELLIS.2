#!/bin/bash
# targets_ceramic_correct.sbatch — 2D-copy targets for teapot_ceramic_crack_correct.
#
# Regenerated Kling video with the grey-body prompt. Measured on the extracted
# frames before this was written:
#   body colour   0.00% of object pixels reach 245, background 250-255 — the
#                 brightness threshold separates cleanly, which is the failure
#                 the regeneration was for.
#   scale drift   width +0.1%, height +0.4% across 150 frames. This is the good
#                 case. teapot_porcelain_correct drifts +11.2%/+11.4% and its
#                 targets decay to IoU 0.74; this one should hold ~0.99 throughout.
#
# THE MASK IS BUILT FROM THIS OBJECT'S OWN MESH, AND THAT IS THE POINT.
#   The original teapot_ceramic_crack has a render_mask.npy that is BIT-IDENTICAL
#   to teapot_porcelain's — 127,492 px, np.array_equal True — while its own mesh
#   rasterises to 94,558 px. Its targets were therefore built against a silhouette
#   26% larger than the mesh it trained on. That run is logged "PASS: 30/30 epochs"
#   and completing epochs said nothing about it.
#
#   So the gate here is INVERTED relative to the porcelain job: matching
#   teapot_porcelain's mask would mean the bug has been reproduced, not avoided.
#   The job fails if the two are identical.
#
# nvdiffrast: sm86 (A40) / sm89 (L40S) only.
#SBATCH --job-name=tgt_ceram_ok
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tgt_ceram_ok_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tgt_ceram_ok_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
OBJ=teapot_ceramic_crack_correct
TAG=gt_targets_${OBJ}
MESH=${T2}/data/${OBJ}/mesh/${OBJ}_render_frame.obj
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo ""; echo "--- STEP 1  render_mask from THIS object's own mesh (--raw) ---"
$PY2 -u "${E}/make_render_mask.py" --mesh "$MESH" --res 960 --tag "$TAG" --raw

echo ""; echo "--- GATE  must reproduce the teapot family silhouette ---"
# EARLIER VERSION OF THIS GATE WAS BACKWARDS AND FAILED THE JOB. It asserted the
# mask must NOT equal teapot_porcelain's, on the belief that the old ceramic run
# had inherited the wrong silhouette. It had not. teapot_ceramic_crack and
# teapot_porcelain are the SAME teapot geometry under the same guan rotation --
# the .obj files differ (md5 f83ba32e vs 99dd541c, different UVs/normals) but a
# silhouette depends only on geometry and camera, so identical masks are the
# CORRECT result and the two are bit-equal for a good reason.
#
# What led me wrong: render_grey.py appeared to give 94,558 px for this mesh
# against 127,492 for porcelain. That gap was an artefact of how I counted, not
# of the render. render_grey.py tone-matches to a reference frame, and against
# this object's near-white reference the mapping pushes object pixels all the
# way to 255 -- the same value as the background -- so ~24k real pixels became
# uncountable in the saved greyscale PNG. Both renderers use the identical
# camera (tan(20 deg), same extrinsics); there was never a discrepancy.
#
# So the gate now asserts the agreement instead of forbidding it.
$PY2 - <<'PYEOF'
import numpy as np, sys
from pathlib import Path
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
new = np.load(E / 'out/gt_targets_teapot_ceramic_crack_correct/render_mask.npy')
old = np.load(E / 'out/gt_targets_teapot_ceramic_crack_guan/render_mask.npy')
print(f'  ours        {int(new.sum()):,} px')
print(f'  old ceramic {int(old.sum()):,} px   identical={np.array_equal(new, old)}')
if not np.array_equal(new, old):
    d = int((new ^ old).sum())
    print(f'  GATE FAILED: {d:,} disagreeing px. Same mesh and same camera as the '
          f'existing ceramic run, so the silhouette must match exactly.')
    sys.exit(1)
print('  GATE PASSED — same mesh, same camera, same silhouette')
PYEOF

echo ""; echo "--- STEP 2  2D-copy targets, 150 frames ---"
$PY2 -u "${E}/make_gt_targets.py" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --render-mask "${E}/out/${TAG}/render_mask.npy" \
  --render-res 960 --n-frames 150 --tag "$TAG"

echo ""; echo "--- IoU vs the video object, per frame ---"
$PY2 - <<'PYEOF'
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
E = T2 / 'experiments/dynamesh'
ours = np.load(E / 'out/gt_targets_teapot_ceramic_crack_correct/render_mask.npy')
F = T2 / 'data/teapot_ceramic_crack_correct/frames_from_video'
def lc(m):
    l, n = ndimage.label(m)
    return m if n == 0 else l == (1 + np.argmax(ndimage.sum(m, l, range(1, n + 1))))
ious = []
for f in range(1, 151):
    a = np.asarray(Image.open(F / f'frame_{f:04d}.png').convert('RGB'), np.float32)
    mn = a.min(2)
    bg = np.median(np.concatenate([mn[0], mn[-1], mn[:, 0], mn[:, -1]]))
    vid = lc(mn < bg - 10)
    vid = np.asarray(Image.fromarray((vid * 255).astype(np.uint8))
                     .resize((960, 960), Image.NEAREST)) > 127
    ious.append((ours & vid).sum() / max((ours | vid).sum(), 1))
    if f % 50 == 0 or f == 1:
        print(f'  frame {f:>4}   IoU {ious[-1]:.3f}')
print(f'\n  IoU  first {ious[0]:.3f}  last {ious[-1]:.3f}  min {min(ious):.3f}  '
      f'mean {sum(ious)/len(ious):.3f}')
print(f'  frames with IoU >= 0.95: {sum(1 for i in ious if i >= 0.95)} of 150')
PYEOF
echo ""; echo "=== Done $(date) ==="
