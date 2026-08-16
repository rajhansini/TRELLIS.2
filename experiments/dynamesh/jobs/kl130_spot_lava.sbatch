#!/bin/bash
# rung20 + L1 on spot_lava.
#
# ONE change from the rung20 L2 run: --recon l1. The per-pixel penalty becomes
# |render-target| instead of (render-target)^2. Mask, trust map, normalisation,
# reduction, optimiser, schedule, seed: all identical.
#
# WHY: l2 is mean-seeking -- at a pixel the model is unsure about, the minimiser
# is the average of the plausible values, and uncertainty peaks at texture
# boundaries, so the hedging lands exactly on edges. Measured on teapot_lava2:
# interior edge detail 5.44 vs the target's 9.82 WITH full supervision. l1's
# minimiser is the median, which commits. 3DGS uses l1 for this reason.
#
# EXPECT PSNR TO FALL. PSNR = 10*log10(1/MSE) and this run no longer minimises
# MSE. Judge on edge detail, not on the headline number.
#
# Verified before submission: recon enters the config hash only when != l2, so
# the l2 directories are untouched; run id pre-computed, new and unique.
#SBATCH --job-name=kl130_spot
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/kl130_spot_lava_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/kl130_spot_lava_%j.log
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
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/mesh/spot_render_frame.obj" \
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_spot_lava/frames" \
  --render-res 960 --resolution 512 \
  --targets kv --blocks all --rank 4 \
  --recon l1 \
  --conf-weight --conf-tau 0.25 --conf-quantile -1 \
  --kl --kl-blocks all --w-kl 130 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
