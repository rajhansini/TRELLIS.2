#!/bin/bash
# SMOKE: rung20 + L1 + LPIPS on teapot_lava2. 2 epochs x 8 frames.
# Purpose is NOT a result. Three questions:
#   1. does the LPIPS term construct and run inside the training graph
#   2. do the LoRA gradients survive a second loss with a different magnitude
#      (the fp16 decoder needs LOSS_SCALE=4096 to keep them alive at all)
#   3. what is the RATIO of w*lpips to the data term -- that number sets w_lpips
#SBATCH --job-name=smoke_l1lp
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/smoke_l1lp_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/smoke_l1lp_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

${PY2} -u "${E}/rung20_conf_lora.py" \
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply" \
  --gt-dir "${T2}/data/teapot_lava2/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_teapot_lava2/frames" \
  --render-res 960 --resolution 512 \
  --targets kv --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --conf-weight --conf-tau 0.25 --conf-quantile -1 \
  --seed 42 --lr 1e-4 \
  --smoke \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
