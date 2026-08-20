#!/bin/bash
# TRELLIS 1 arm of the three-mode conditioning comparison.
# step06b_asset.py, phase d = window [t-1,t,t+1]; --betas 0.0 so NO enhancement
# bias is applied and the run is pure frozen TRELLIS + the blend.
# Geometry is generated from a reference frame; no mesh is supplied.
# Pass OBJ, VARIANT (v2|v2b|v3), NFR.
#SBATCH --job-name=ta1
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta1_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta1_%j.log
set -eo pipefail
T1=/net/projects/ranalab/rajhansini/TRELLIS
PY1=/net/projects/ranalab/rajhansini/conda_envs/trellis/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis/bin:$PATH"
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export ATTN_BACKEND=xformers

# PRIVATE torch.hub cache per job.
# trellis_image_to_3d.py calls torch.hub.load('facebookresearch/dinov2'), and
# torch.hub's _get_cache_or_reload() rmtree's the repo dir before re-extracting.
# Twelve jobs launched together all did that to the SAME ~/.cache/torch/hub and
# every one of them died with "OSError: [Errno 39] Directory not empty".
# The repo is 3.6 MB, so each job gets its own copy; the WEIGHTS (2.1 GB, incl.
# dinov2_vitl14_reg4_pretrain.pth) are symlinked read-only from scratch rather
# than copied, and torch.hub never deletes checkpoints/.
export TORCH_HOME=/tmp/torchhub_${SLURM_JOB_ID}
_SRC=/net/scratch/rajhansini/.cache/torch/hub
mkdir -p "${TORCH_HOME}/hub"
cp -r "${_SRC}/facebookresearch_dinov2_main" "${TORCH_HOME}/hub/" 2>/dev/null || true
ln -sfn "${_SRC}/checkpoints" "${TORCH_HOME}/hub/checkpoints"
: > "${TORCH_HOME}/hub/trusted_list"
trap 'rm -rf "${TORCH_HOME}"' EXIT
echo "TORCH_HOME=${TORCH_HOME} (private; weights symlinked from scratch)"
cd ${T1}/experiments/enhancement
echo "=== $(date) host=$(hostname) OBJ=${OBJ} VARIANT=${VARIANT} phase=d ==="; nvidia-smi -L
${PY1} -u step06b_asset.py \
  --asset "ta1_${OBJ}_${VARIANT}" \
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/${OBJ}/frames_from_video" \
  --phase d --variant "${VARIANT}" --betas 0.0 \
  --start_frame 1 --frames "${NFR}" --base_seed 6 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== done $(date) ==="
