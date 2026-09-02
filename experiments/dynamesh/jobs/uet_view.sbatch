#!/bin/bash
# Fixed-camera render of a BATCH H run at an arbitrary yaw.
#
# jobs/fixview_arm.sbatch pins --turns 0 but leaves --yaw0 at 0, so it can only produce the
# supervised view. Fig. 7's layout needs a second, different camera, hence this variant.
#
# YAW is RELATIVE to the baked mesh. unicorn_extended_texture_render_frame.obj already
# carries R(0,0)^T.R(20,12), so YAW=0 here IS the original yaw 20 / elev 12 the Kling clip
# was shot from, and any YAW is an offset on top of that.
#
# --sweep both (not angle): an MCFM checkpoint blends a WINDOW of conds and indexes
# _raws[0..n-1], so --sweep angle walks off its 1-element list with an IndexError.
#SBATCH --job-name=uetv
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
set -eo pipefail
: "${RUN:?}"; : "${TAG:?}"; : "${YAW:?}"; ELEV=${ELEV:-0}; MESH=${MESH:-}
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) RUN=${RUN} YAW=${YAW} ELEV=${ELEV} TAG=${TAG} ==="; nvidia-smi -L
${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" --sweep both ${MESH:+--mesh "${MESH}"} \
  --n-frames 150 --n-angles 150 --turns 0 --yaw0 "${YAW}" --elev "${ELEV}" \
  --radius 2.0 --res 518 --tag "uet_${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
N=$(ls "${E}/out/uet_${TAG}/frames"/*.png 2>/dev/null | wc -l)
[ "$N" -eq 150 ] || { echo "FAILED: ${N} frames, expected 150"; exit 1; }
echo "wrote ${N} frames to ${E}/out/uet_${TAG}/frames"
