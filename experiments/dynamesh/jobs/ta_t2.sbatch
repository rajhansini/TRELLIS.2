#!/bin/bash
# TRELLIS.2 arm of the three-mode conditioning comparison.
# FROZEN model, no adapter (--frozen-only). The blend is the only thing that
# differs between arms; mesh, noise, camera and geometry are all pinned.
# Pass RUN, OBJ, MODE, NFR.
#SBATCH --job-name=ta2
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/ta2_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) OBJ=${OBJ} MODE=${MODE} frozen ==="; nvidia-smi -L
${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --yaw0 0 --elev 0 \
  --frozen-only --mcfm "${MODE}" --res 518 \
  --tag "ta2_${OBJ}_${MODE}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== done $(date) ==="
