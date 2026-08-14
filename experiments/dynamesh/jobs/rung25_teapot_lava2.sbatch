#!/bin/bash
# rung25 on teapot_lava2 — LoRA target set = kvo (to_kv + to_out), NO to_q.
#
# WHY THIS RUN EXISTS: kvo is the highest number in the project (23.37 dB) but
# only on the ORIGINAL teapot at 518^2, where PSNR is scored over 28,559 px
# instead of 98,083 — not comparable to anything in the current grid. rung19
# (qkvo) tests to_q + to_kv + to_out together; without this run there is no way
# to tell whether a rung19 gain came from to_q or from to_out.
#
# Verified before submission by executing the script's own config code:
#   rung=25  targets=('to_kv','to_out')  params=860,160  B matrices=60
#   conf_weight=False   run id new, no collision
# to_out was exercised and gated in smoke 2177793 (qkvo, 90 B matrices,
# GATE-plain 0.000e+00); to_kv has run 12 times in the current grid.
#SBATCH --job-name=r25_teap
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r25_teap_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r25_teap_%j.log
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
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/teapot_lava2/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_teapot_lava2/frames" \
  --render-res 960 --resolution 512 \
  --targets kvo --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
