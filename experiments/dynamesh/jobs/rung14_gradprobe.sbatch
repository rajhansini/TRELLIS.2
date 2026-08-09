#!/bin/bash
# rung14 gradient probe: does a gradient reach all 90 cross-attention LoRA
# matrices through TRELLIS.2's bf16 1.3B texture flow?
#SBATCH --job-name=r14_grad
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14grad_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14grad_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$ROOT/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="; nvidia-smi -L
$PY2 -u "$E/rung14_crossattn_lora.py" \
  --mesh "$ROOT/render/out/f0075/frozen_f0075.ply" \
  --rank 4 --targets cross --resolution 1024 --smoke \
  2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|^\s+@|^\s+_TORCH|^\s+def |it/s|%\|"
echo "=== Done $(date) ==="
