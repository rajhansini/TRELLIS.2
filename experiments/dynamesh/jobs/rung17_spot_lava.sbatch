#!/bin/bash
# rung17 on spot_lava — IDENTICAL to its rung20 run except the trust map is OFF.
#
# rung17 vs rung20 is ONE FLAG. Omitting --conf-weight makes _RUNG fall through
# to {'kv': 17}, and conf_w stays all-ones, so every pixel in the silhouette is
# weighted equally instead of by cos(n,v)/footprint with a hard cutoff at 0.25.
# Nothing else differs: same mesh, same video frames, same targets, same rank,
# same seed, same lr, same epochs, same render resolution.
#
# Targets already exist and are NOT rebuilt — verified before submission:
#   gt_targets_spot_lava/frames  150 gt_*.png  150 valid_*.npy   render_mask.npy 960^2
# Run id pre-computed with the script's own config code; new directory, no collision.
#SBATCH --job-name=r17_spotlava
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 4 h is the partition cap; checkpoints every epoch and --requeue resumes,
# guarded by GATE-resume on mesh / targets / render_res.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r17_spotlava_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r17_spotlava_%j.log
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
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/mesh/spot_render_frame.obj" \
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_spot_lava/frames" \
  --render-res 960 --resolution 512 \
  --targets kv --blocks all --rank 4 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
