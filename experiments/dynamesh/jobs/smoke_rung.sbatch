#!/bin/bash
# SMOKE: 2 epochs, 8 frames -- does this rung RUN AT ALL?
# Static analysis cannot catch a shape mismatch when the context is 3x longer, a
# dtype error in the KL on the dual branch, or an OOM. rung32/33/34 have never
# executed a single step; this is the cheapest thing that proves they can.
# Pass SCRIPT and EXTRA.
#SBATCH --job-name=smoke
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/smoke_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/smoke_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}
echo "=== SMOKE $(date) host=$(hostname) SCRIPT=${SCRIPT} EXTRA=${EXTRA} ==="; nvidia-smi -L
${PY2} -u "${E}/${SCRIPT}" \
  --mesh "${T2}/data/spot_lava/mesh/spot_render_frame.obj" \
  --gt-dir "${T2}/data/spot_lava/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_spot_lava/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 --smoke \
  ${EXTRA} 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== SMOKE DONE $(date) ==="
