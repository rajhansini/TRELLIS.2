#!/bin/bash
# Per object, end to end: backprojection (option C) -> 3-panel video.
#   1 GT video frame  |  2 2D copy (option B)  |  3 backprojection (option C)
# One object per job. Pass OBJ, MESH, GTDIR, NFR, BGT.
#SBATCH --job-name=gt3p
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt3p_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt3p_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd ${T2}
echo "=== $(date) host=$(hostname) OBJ=${OBJ} nfr=${NFR} bg=${BGT} ==="; nvidia-smi -L

echo "--- 1/3 backproject (option C: image -> vertices) ---"
${PY2} -u "${E}/backproject_gt.py" --mesh "${MESH}" --gt-dir "${GTDIR}" \
  --n-frames "${NFR}" --raster-res 960 --bg-thresh "${BGT}" \
  --tag "gt_backproj_${OBJ}" 2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn"

echo "--- 2/3 re-render backprojected vertex colours ---"
${PY2} -u "${E}/render_backproj_gt.py" --mesh "${MESH}" \
  --targets-npz "${E}/out/gt_backproj_${OBJ}/targets.npz" \
  --render-res 960 --n-frames "${NFR}" --tag "gt_rendered_${OBJ}" 2>&1 \
  | grep -vE "FutureWarning|UserWarning|warnings.warn"

echo "--- 3/3 three-panel video ---"
${PY2} -u "${E}/render_gt_vs_target_video.py" --obj "${OBJ}" 2>&1 \
  | grep -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
