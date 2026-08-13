#!/bin/bash
# rung23 — RANK SWEEP over the rung22 method (magnitude ceiling, no pairwise term).
#
# Everything is held identical to rung22 except --rank. alpha is FIXED at 4 across
# every arm: dW is scaled by alpha/rank (Hu et al. 2021 sec 4.1), so the sweep
# measures rank alone rather than rank confounded with effective step size. At
# rank 4 the factor is exactly 1.0, so the r=4 arm IS rung22 — no re-run needed.
#
# Launch:  RANK=8 sbatch jobs/rung23_ranksweep.sbatch
#SBATCH --job-name=r23sweep
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# NO h100/h200. nvdiffrast in conda_envs/trellis2 has no kernel image for the
# sm_90-class cards on h002: RasterizeCudaContext() dies with
#   Cuda error: 209[cudaFuncGetAttributes(&attr, (void*)fineRasterKernel)]
# = cudaErrorNoKernelImageForDevice. Ranks 16 and 32 were lost to this after 48s
# (jobs 2164472/2164473) purely because the scheduler put them on h002 — nothing
# to do with rank. Rebuild nvdiffrast for sm_90 to lift this restriction.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r23sweep_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r23sweep_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2

if [ -z "${RANK}" ]; then echo "FATAL: RANK is unset — refusing to silently run rank 4"; exit 1; fi
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} RANK=${RANK} ALPHA=${ALPHA:-4} ==="
nvidia-smi -L

${PY2} -u "${E}/rung17_backproj_lora.py" \
  --mesh /net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply \
  --targets kv --blocks all --resolution 512 --render-res 518 \
  --rank "${RANK}" --lora-alpha "${ALPHA:-4}" \
  --conf-weight --conf-tau 0.25 --conf-quantile -1 \
  --lambda-con "${LCON:-0.1}" --lambda-smo 0 --reg-k-voxels 2 \
  --lambda-smooth 0 \
  --epochs 30 --n-frames 150 --seed 42 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
