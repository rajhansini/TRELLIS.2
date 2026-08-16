#!/bin/bash
#SBATCH --job-name=r30obj
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r30obj_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r30obj_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) OBJ=${OBJ} MESH=${MESH} ==="; nvidia-smi -L

for spec in side:0 360:1 720:2; do
  V=${spec%%:*}; TN=${spec##*:}
  echo ""; echo "### ${OBJ} ${V} (turns=${TN})"
  ${PY2} -u "${E}/render_rung30_transfer.py" \
    --run "${RUN}" --ckpt lora_best.pt --mesh "${MESH}" \
    --sweep both --n-frames 150 --res 720 --turns "${TN}" --fps 20 \
    --tag "r30h_${OBJ}_${V}" \
    2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH"
done
echo "=== Done $(date) ==="
