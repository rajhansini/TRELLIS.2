#!/bin/bash
# rung17 + L1 + LPIPS on spot_lava.
# Perceptual weight 0.1 -- MEASURED on smoke job 2176443, where it came to
# exactly 10% of the total loss (l1 0.247, lpips 0.286, w*lp 0.029) and peak
# memory was 17.6 GiB, BELOW the 19.7 GiB of the L2 runs. Not a guess.
#SBATCH --job-name=k17_4000_spot
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/k17_4000_spot_lava_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/k17_4000_spot_lava_%j.log
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
  --recon l1 --lpips --w-lpips 0.1 \
  --kl --kl-blocks all --w-kl 4000 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
