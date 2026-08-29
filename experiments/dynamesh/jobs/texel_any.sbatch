#!/bin/bash
# Texel temporal metrics for ONE (object, arm). Pass RUN, TAG, NFR, SCRIPT.
#
# RENDERER DISPATCH IS NOT COSMETIC. Verified from the checkpoints themselves:
# rungs 27/28/29/30/32/34 carry 300 tensors and no gates; rungs 31/33 carry 510
# tensors plus 30 per-block gates (the dual cross-attention branch). Loading a
# 510-tensor checkpoint into the 300-tensor registry raises "Unexpected key(s)";
# the reverse loads CLEANLY and silently reports metrics for a model that never
# ran. So SCRIPT is passed in by the submitter from the measured structure, not
# guessed from the rung number.
#
# --skip-render means no PNGs and no ffmpeg: this is the ODE decode only, writing
# one JSON. Metrics are computed on the PBR voxel field, so they are
# view-independent and this needs no camera argument beyond the fixed default.
#SBATCH --job-name=tex
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tex_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/tex_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
: "${RUN:?}"; : "${TAG:?}"; : "${NFR:?}"; : "${SCRIPT:?}"
case "$RUN" in /*) ;; *) RUN="${E}/${RUN}";; esac
[ -f "${RUN}/config.json" ] || { echo "no config.json under ${RUN}"; exit 1; }
mkdir -p "${E}/out/TEXEL"
OUTJ="${E}/out/TEXEL/${TAG}.json"
if [ -s "$OUTJ" ]; then echo "ALREADY DONE: $OUTJ"; exit 0; fi
echo "=== $(date) host=$(hostname) TAG=${TAG} SCRIPT=${SCRIPT} ==="; nvidia-smi -L
${PY2} -u "${E}/${SCRIPT}" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --skip-render \
  --texel-metrics "${OUTJ}" --tag "texel_${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
[ -s "$OUTJ" ] || { echo "FAILED: no json written"; exit 1; }
echo "=== Done $(date) ==="
