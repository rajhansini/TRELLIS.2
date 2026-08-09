#!/bin/bash
# TRELLIS.2 per-frame texturing baseline. ONE fixed mesh, N video frames.
# Must run on a compute node: o_voxel's compiled extension needs GLIBC 2.32
# and the login node is 2.31 (compute nodes are 2.35).
#SBATCH --job-name=r14t2_orb
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:L40S:1   # 46 GB; renders peak ~17 GB. A40s are contended,
#                          # cluster mixes 10.5 GiB 2080 Ti with 44.5 GiB A40.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14t2orb_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r14t2orb_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${ROOT}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flex_gemm writes its Triton autotune cache to ~/.flex_gemm/autotune_cache.json
# via a .tmp + rename. Two concurrent jobs race on that rename and one dies with
# FileNotFoundError on the .tmp. Give each job its own copy, seeded from the
# shared one so nothing is re-autotuned.
# (flex_gemm/__init__.py:4 reads FLEX_GEMM_AUTOTUNE_CACHE_PATH.)
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L; echo "glibc: $(ldd --version | head -1)"

${PY2} -u "${E}/render_rung14_t2_orbit.py" \

  --pin-frame 75 --n-angles 120 --tag rung14_t2_orbit \

  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
