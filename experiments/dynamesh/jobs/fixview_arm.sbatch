#!/bin/bash
# FIXED-CAMERA frame export for the video-space comparison metrics.
#
# --turns 0 pins the camera and steps the frame index, so the ONLY thing that moves
# between frames is the texture. An orbit render would make every frame differ from
# the last regardless of the texture, which is the confound video_metrics.py cannot
# see and cannot correct for.
#
# One job renders BOTH arms: render_rung27_orbit.py composites [frozen | ours] into
# a single canvas per frame, which guarantees the two arms share a camera exactly.
# video_metrics.py --panel 0/1 --n-panels 2 crops them back apart and asserts the
# layout arithmetic rather than trusting it.
#
# RES matches whatever the baselines were rendered at. out/frozen_t2/renders is
# 518x518, so RES=518 keeps ours directly comparable to the baseline already on disk.
#SBATCH --job-name=fxv
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fxv_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fxv_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
: "${RUN:?}"; : "${TAG:?}"; : "${NFR:?}"; : "${RES:?}"
case "$RUN" in /*) ;; *) RUN="${E}/${RUN}";; esac
[ -f "${RUN}/config.json" ] || { echo "no config.json under ${RUN}"; exit 1; }
echo "=== $(date) host=$(hostname) RUN=${RUN} TAG=${TAG} nfr=${NFR} RES=${RES} ==="; nvidia-smi -L
${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --res "${RES}" \
  --tag "fixview_${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
N=$(ls "${E}/out/fixview_${TAG}/frames"/*.png 2>/dev/null | wc -l)
[ "$N" -eq "${NFR}" ] || { echo "FAILED: ${N} frames written, expected ${NFR}"; exit 1; }
echo "wrote ${N} frames to ${E}/out/fixview_${TAG}/frames"
echo "=== Done $(date) ==="
