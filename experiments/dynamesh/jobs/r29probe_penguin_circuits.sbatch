#!/bin/bash
#SBATCH --job-name=r29p_penguin_circuits
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r29probe_penguin_circuits_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r29probe_penguin_circuits_%j.log
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
${PY2} -u "${E}/rung29_kl_cross_attention.py" \
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS.2/data/penguin_circuits/mesh/penguin_circuits_render_frame_guan.obj" \
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/penguin_circuits/frames_from_video" \
  --gt-render-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gt_targets_penguin_circuits_guan/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --kl --kl-blocks all --kl-probe --w-kl 0 \
  --epochs 6 --n-frames 121 --seed 42 --lr 1e-4 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward|it/s\]"
echo "=== Done $(date) ==="
