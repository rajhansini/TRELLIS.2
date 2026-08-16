#!/bin/bash
#SBATCH --job-name=r31smoke
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31smoke_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31smoke_%j.log
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

${PY2} -u "${E}/rung31_warmstart_lora.py" \
  --mesh "$T/data/penguin_circuits/mesh/penguin_circuits_render_frame_guan.obj" \
  --gt-dir "$T/data/penguin_circuits/frames_from_video" \
  --gt-render-dir "$E/out/gt_targets_penguin_circuits_guan/frames" \
  --crop-from-mask \
  --init-from "$INIT" \
  --targets qkvo+sa --rank 4 --lora-alpha 4 \
  --recon l1 --lpips \
  --render-res 960 --resolution 512 \
  --n-frames 8 --epochs 2 --seed 42 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH"
echo "=== Done $(date) ==="
