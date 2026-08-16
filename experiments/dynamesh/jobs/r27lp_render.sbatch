#!/bin/bash
# Render one LPIPS-sweep arm at one camera treatment.
#   RUN=<run dir name>   TURNS=<0|1|2>   TAG=<output tag>
#     TURNS 0 = single/side  camera FIXED at the training view, frames 1..121
#     TURNS 1 = 360          one revolution across the sequence
#     TURNS 2 = 720          two revolutions, every surface point seen twice
# Panels: GT | frozen TRELLIS.2 | adapted.
#SBATCH --job-name=r27lprender
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27lprender_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27lprender_%x_%j.log
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
echo "=== $(date) host=$(hostname) RUN=${RUN} TURNS=${TURNS} TAG=${TAG} ==="
nvidia-smi -L

${PY2} -u "${E}/render_rung27_orbit.py" \
  --run "${E}/runs/${RUN}" \
  --ckpt lora_best.pt \
  --mesh "${T2}/data/horse_metal/mesh/horse_metal_render_frame_guan.obj" \
  --sweep both --turns ${TURNS} \
  --n-frames 121 --n-angles 121 \
  --tag "${TAG}" \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH"

echo "=== Done $(date) ==="
