#!/bin/bash
# Temporal Flickering on TEXELS (the PBR voxel field), not on rendered pixels.
# usage: RUN=<rundir> TAG=<name> NFR=<n> sbatch jobs/texel_metric.sbatch
#SBATCH --job-name=texel
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/texel_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/texel_%x_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) RUN=${RUN} ==="; nvidia-smi -L
${PY2} -u "${E}/render_rung31_orbit.py" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --skip-render \
  --texel-metrics "${E}/out/TEXEL/${TAG}.json" --tag "texel_${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
