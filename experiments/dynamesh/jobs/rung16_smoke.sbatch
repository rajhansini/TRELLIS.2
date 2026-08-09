#!/bin/bash
# rung14 SMOKE — 2 epochs on 8 frames. Proves the whole path end to end (gates,
# ODE prefix, graded eval, decode, differentiable render, backward, checkpoint,
# eval, resume) before any real GPU time is committed to it.
#SBATCH --job-name=r16_smoke
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1   # PIN the GPU type: results are memory-sensitive and the
#                          # cluster mixes 10.5 GiB 2080 Ti with 44.5 GiB A40.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r16smoke_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r16smoke_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${ROOT}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
export SPCONV_ALGO=native
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L; echo "glibc: $(ldd --version | head -1)"

${PY2} -u "${E}/rung16_intersection_lora.py" --mesh /net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply --smoke --render-res 512 --seed 42 --resolution 512 --targets "${TARGETS:-kv}" --blocks "${BLOCKS:-all}" 2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
