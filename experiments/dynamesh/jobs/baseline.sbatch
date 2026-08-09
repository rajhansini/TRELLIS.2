#!/bin/bash
# TRELLIS.2 per-frame texturing baseline. ONE fixed mesh, N video frames.
# Must run on a compute node: o_voxel's compiled extension needs GLIBC 2.32
# and the login node is 2.31 (compute nodes are 2.35).
#SBATCH --job-name=t2_baseline
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1   # PIN the GPU type: results are memory-sensitive and the
#                          # cluster mixes 10.5 GiB 2080 Ti with 44.5 GiB A40.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/baseline_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/baseline_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${ROOT}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L; echo "glibc: $(ldd --version | head -1)"

${PY2} -u "${E}/run_baseline.py" \
  --mesh "${MESH:-/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075_uv.obj}" \
  --n-frames "${NFRAMES:-150}" --tag "${TAG:-frozen_t2}" --seed 42 \
  --render-res 518 \
  2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
