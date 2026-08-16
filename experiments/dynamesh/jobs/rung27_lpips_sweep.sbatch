#!/bin/bash
# rung27 (qkvo+sa) LPIPS-WEIGHT SWEEP — horse_metal.
#
# Answers the reviewer question "why w_lpips = 0.1?" by sweeping it and reporting
# the sweep. Precedent: TRELLIS uses 0.2 for LPIPS in its Gaussian/RF decoder
# (L_recon = L1 + 0.2(1-SSIM) + 0.2 LPIPS) with no stated justification, and the
# flow model it trains with pure CFM velocity MSE uses no perceptual term at all.
# So 0.2 is included explicitly as the one referenced value on the axis.
#
#   WLP=<0.05|0.1|0.2|0.5>   or   NOLPIPS=1 for the w=0 arm
#
# The w=0 arm passes NO --lpips at all rather than --w-lpips 0: that skips loading
# AlexNet entirely and is the honest "no perceptual term" control. It also hashes
# differently, so it cannot land in a weighted run's directory.
#
# GPU: a40|L40S only (sm86/sm89 kernel images). Mesh + GT targets are the _guan
# pair, matching the existing horse_metal job.
#SBATCH --job-name=r27lp
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27lp_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27lp_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} w_lpips=${WLP:-OFF} ==="
nvidia-smi -L

${PY2} -u "${E}/rung27_selfattn_lora.py" \
  --mesh "${T2}/data/horse_metal/mesh/horse_metal_render_frame_guan.obj" \
  --gt-dir "${T2}/data/horse_metal/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_horse_metal_guan/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 \
  ${NOLPIPS:+} ${WLP:+--lpips --w-lpips ${WLP}} \
  --epochs 30 --n-frames 121 --seed 42 --lr 1e-4 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
