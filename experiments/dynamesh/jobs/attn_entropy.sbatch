#!/bin/bash
# DIAGNOSTIC, trains nothing. Measures cross-attention entropy per voxel, split
# by whether the training camera sees that voxel. --out-dir points at an existing
# trained run so its lora_best.pt is loaded and its config reproduced.
#SBATCH --job-name=attn_ent
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/attn_ent_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/attn_ent_%j.log
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

${PY2} -u "${E}/kl_attn_lora.py" \
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply" \
  --gt-dir "${T2}/data/teapot_lava2/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_teapot_lava2/frames" \
  --render-res 960 --resolution 512 \
  --targets kv --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  --out-dir "${E}/runs/rung17_l1_lp_all_kv_r4_s42_37676d82" \
  --attn-entropy \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
