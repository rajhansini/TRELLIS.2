#!/bin/bash
#SBATCH --job-name=frzprobe
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frzprobe_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frzprobe_%j.log
set -eo pipefail
T=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd $T
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
# FROZEN ONLY. No adapter is loaded into the forward at all (--frozen-only),
# each arm conditioned on its OWN object's video. Question: does frozen
# TRELLIS.2 produce a black object regardless of how bright the input is?
${PY2} -u "${E}/render_rung30_transfer.py" \
  --run "${RUN}" --ckpt lora_best.pt \
  --mesh "${MESH}" --cond-dir "${COND}" \
  --frozen-only --allow-same-mesh \
  --sweep angle --pin-frame 1 --n-angles 24 --res 720 --fps 8 \
  --tag "${TAG}" \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH"
echo "=== Done $(date) ==="
