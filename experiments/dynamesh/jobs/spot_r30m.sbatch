#!/bin/bash
# The one cell missing from the 336: spot_lava rung30 + temporal-only MCFM at the
# CANONICAL beta pair (0.957529 / 0.293824) used by every other object.
#
# spot_lava has sixteen rung30+M runs already, but every one is a cell of its own
# 4x4 KL grid at per-object calibrated betas, so none is comparable with the other
# 23 objects. Averaging a grid cell into the table would silently mix two
# calibration protocols in one row.
#SBATCH --job-name=spot_r30m
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/spot_r30m_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/spot_r30m_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}; echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
${PY2} -u "${E}/rung30_kl_both.py" \
  --mesh "${T2}/data/spot_lava/mesh/spot_render_frame.obj" \
  --gt-dir "${T2}/data/spot_lava/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_spot_lava/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --mcfm v2_D --kl --w-kl 0.957529 --w-kl-self 0.293824 \
  --kl-blocks all --kl-voxels 256 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
