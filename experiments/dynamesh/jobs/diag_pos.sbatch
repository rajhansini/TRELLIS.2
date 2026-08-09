#!/bin/bash
#SBATCH --job-name=diagpos
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/diagpos_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/diagpos_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPCONV_ALGO=native
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
${PY2} -u "${E}/diag_pos.py" 2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|it/s\]$"
echo "=== Done $(date) ==="
