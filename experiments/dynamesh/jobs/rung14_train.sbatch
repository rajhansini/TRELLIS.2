#!/bin/bash
#SBATCH --job-name=r14_train
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1   # PIN the GPU type: results are memory-sensitive and the
#                          # cluster mixes 10.5 GiB 2080 Ti with 44.5 GiB A40.
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14train_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14train_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$ROOT/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi -L
$PY2 -u "$E/rung14_crossattn_lora.py" \
  --mesh "$ROOT/render/out/f0075/frozen_f0075.ply" \
  --rank 4 --targets cross --resolution "${RES:-512}" \
  --epochs ${EPOCHS:-30} --n-frames ${NFRAMES:-150} \
  2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|^\s+@|^\s+_TORCH|^\s+def |it/s|%\|"
echo "=== Done $(date) ==="
