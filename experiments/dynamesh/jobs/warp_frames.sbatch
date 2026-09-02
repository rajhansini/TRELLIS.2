#!/bin/bash
# One object per job: bake the solved rotation and warp that clip's frames onto it.
#SBATCH --job-name=warp
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/warp_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/warp_%x_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd ${T2}
: "${NAME:?}"; : "${OBJ:?}"; : "${MESH:?}"
echo "=== $(date) host=$(hostname) NAME=${NAME} OBJ=${OBJ} ${DRY:+DRY-RUN} ==="; nvidia-smi -L
${PY2} -u "${E}/warp_frames_to_render.py" --name "${NAME}" --obj "${OBJ}" --mesh "${MESH}" ${DRY:+--dry-run} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
