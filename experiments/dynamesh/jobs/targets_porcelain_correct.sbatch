#!/bin/bash
# targets_porcelain_correct.sbatch — 2D-copy targets for teapot_porcelain_correct.
#
# The regenerated Kling video, with the prompt fixed so the body no longer turns
# white. Measured on all 150 frames: 0.00% of object pixels reach 245 in the
# min-channel, against a background of 249-255, so the brightness threshold that
# make_gt_targets.py uses separates cleanly. That was the failure this video was
# regenerated to fix, and it is fixed.
#
# WHAT IS STILL WRONG, AND WHY THIS RUNS ANYWAY
#   The teapot SCALES. Bounding box 998->1110 px wide and 559->623 tall over the
#   150 frames, +11.2% and +11.4% — isotropic growth, not a threshold artefact,
#   measured directly on the silhouette. teapot_lava2, the reference good case,
#   drifts 0.42%. 2D copy assumes the video object and our fixed mesh render share
#   a silhouette; an 11% linear drift breaks that assumption late in the sequence,
#   so the per-pixel loss there becomes partly a shape comparison rather than a
#   texture one. The targets are built regardless so the damage can be SEEN and
#   the usable prefix chosen from data, rather than argued about.
#
#   Within  1% width drift: frames 1-28
#   Within  2%:             frames 1-43
#   Within  3%:             frames 1-54
#   Within  5%:             frames 1-77
#
# MESH AND RESOLUTION
#   The mesh is teapot_porcelain's guan .obj, already baked into the render frame,
#   so --raw is required exactly as build_targets_new_objects.py documents: that
#   script's preprocess() carries the wrong axis map and --raw skips it.
#   Built at 960^2 to match every other object even though the video is 1440^2 —
#   make_gt_targets.py thresholds at full resolution and only then resizes, so no
#   precision is lost by the mismatch.
#
#   render_mask.npy MUST come out at 127,492 px, bit-identical to
#   gt_targets_teapot_porcelain_guan. Same mesh, same camera, so anything else
#   means the mesh copy is wrong. Checked below, and the job fails if it differs.
#
# nvdiffrast: sm86 (A40) / sm89 (L40S) only.
#SBATCH --job-name=tgt_porc_ok
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tgt_porc_ok_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tgt_porc_ok_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
OBJ=teapot_porcelain_correct
TAG=gt_targets_${OBJ}
MESH=${T2}/data/${OBJ}/mesh/${OBJ}_render_frame.obj
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

echo ""; echo "--- STEP 1  render_mask (--raw, mesh already in camera frame) ---"
$PY2 -u "${E}/make_render_mask.py" --mesh "$MESH" --res 960 --tag "$TAG" --raw

echo ""; echo "--- GATE  silhouette must equal teapot_porcelain's 127,492 px ---"
$PY2 - <<'PYEOF'
import numpy as np, sys
from pathlib import Path
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
new = np.load(E / 'out/gt_targets_teapot_porcelain_correct/render_mask.npy')
ref = np.load(E / 'out/gt_targets_teapot_porcelain_guan/render_mask.npy')
print(f'  new {int(new.sum()):,} px   ref {int(ref.sum()):,} px   '
      f'identical={np.array_equal(new, ref)}')
if not np.array_equal(new, ref):
    d = int((new ^ ref).sum())
    print(f'  GATE FAILED: {d:,} disagreeing px — the mesh copy is not the same '
          f'mesh/camera as teapot_porcelain. Refusing to build targets on it.')
    sys.exit(1)
print('  GATE PASSED')
PYEOF

echo ""; echo "--- STEP 2  2D-copy targets, 150 frames ---"
$PY2 -u "${E}/make_gt_targets.py" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --render-mask "${E}/out/${TAG}/render_mask.npy" \
  --render-res 960 --n-frames 150 --tag "$TAG"

echo ""; echo "--- IoU of our silhouette against the video object, per frame ---"
echo "--- this is where the +11% scale drift shows up ---"
$PY2 - <<'PYEOF'
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
E = T2 / 'experiments/dynamesh'
ours = np.load(E / 'out/gt_targets_teapot_porcelain_correct/render_mask.npy')
F = T2 / 'data/teapot_porcelain_correct/frames_from_video'
def lc(m):
    l, n = ndimage.label(m)
    return m if n == 0 else l == (1 + np.argmax(ndimage.sum(m, l, range(1, n + 1))))
print(f'  {"frame":>6}{"IoU":>9}{"ours outside video":>21}')
rows = []
for f in range(1, 151):
    a = np.asarray(Image.open(F / f'frame_{f:04d}.png').convert('RGB'), np.float32)
    mn = a.min(2)
    bg = np.median(np.concatenate([mn[0], mn[-1], mn[:, 0], mn[:, -1]]))
    vid = lc(mn < bg - 10)
    vid = np.asarray(Image.fromarray((vid * 255).astype(np.uint8))
                     .resize((960, 960), Image.NEAREST)) > 127
    iou = (ours & vid).sum() / max((ours | vid).sum(), 1)
    out = (ours & ~vid).sum() / max(ours.sum(), 1)
    rows.append((f, iou, out))
    if f % 25 == 0 or f == 1:
        print(f'  {f:>6}{iou:>9.3f}{100*out:>20.2f}%')
i = [r[1] for r in rows]
print(f'\n  IoU  first {i[0]:.3f}   last {i[-1]:.3f}   min {min(i):.3f}')
ok = [r[0] for r in rows if r[1] >= 0.95]
print(f'  frames with IoU >= 0.95: {len(ok)} of 150'
      + (f' (1-{max(ok)})' if ok else ''))
PYEOF
echo ""; echo "=== Done $(date) ==="
