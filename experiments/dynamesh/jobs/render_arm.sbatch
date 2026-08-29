#!/bin/bash
# One (object, arm, view) per job. 15 objects x 14 arms x 4 views is 840 independent
# renders -- they must never be looped inside one job.
#
# Pass RUN (ABSOLUTE path), TAG, YAW0, ELEV, NFR.  TURNS defaults to 0: the camera is
# FIXED, so every change between frames is texture. That is required for flicker,
# which is a second temporal difference and would otherwise measure camera motion.
#
# render_arm.py picks the trainer whose registry matches the checkpoint: rungs 31/33
# save 510 tensors with `gates` and cannot load into a rung27 registry.
#SBATCH --job-name=rarm
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"   # nvdiffrast has no kernel for h100/h200 (CUDA 209)
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G      # inference-only (no backward/optimizer state); training peaked
#                      # at 15.3G doing MORE work, so this has 1.5x headroom
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/RENDERS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/RENDERS/%x_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
# Triton defaults its kernel cache to $HOME/.triton, which is NFS. With ~90 of
# these renders in flight at once they read and rewrite the same cache entries
# and hit "OSError: [Errno 116] Stale file handle" mid-compile -- 11 of the
# first 19 jobs of the batch-C fleet died that way. Node-local dir per job, warm
# -started from the NFS copy so kernels are not recompiled from scratch. Same
# shape as the FLEX_GEMM line above.
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
cp -r "$HOME/.triton/cache/." "$TRITON_CACHE_DIR/" 2>/dev/null || true
trap 'rm -rf "$TRITON_CACHE_DIR" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH"' EXIT
mkdir -p ${E}/out/RENDERS
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) TAG=${TAG} yaw0=${YAW0} elev=${ELEV} turns=${TURNS:-0} nfr=${NFR} ==="
echo "    RUN=${RUN}"
nvidia-smi -L

# Pre-flight: an absent config.json means render_arm.py cannot know the rung or the
# mcfm mode. A relative RUN path once killed 60 render jobs this way.
if [ ! -f "${RUN}/config.json" ]; then
  echo "FAILED: ${RUN}/config.json missing — refusing to render an unknown arm"; exit 1
fi

# Idempotent: a requeued job must not redo finished work.
# NOTE: never `ls glob | wc -l` here. Under `set -eo pipefail` a missing frames dir
# makes ls exit 2, pipefail propagates, set -e kills the job -- which silently broke
# every FIRST render of a new object.
HAVE=0
if [ -d "${E}/out/${TAG}/frames" ]; then
  HAVE=$(find "${E}/out/${TAG}/frames" -maxdepth 1 -name "*.png" | wc -l)
fi
if [ "${HAVE}" -ge "${NFR}" ]; then
  echo "ALREADY COMPLETE: ${TAG} has ${HAVE}/${NFR} frames"; exit 0
fi
echo "have ${HAVE}/${NFR} frames, rendering"

${PY2} -u "${E}/render_arm.py" --run "${RUN}" --tag "${TAG}" \
  --yaw0 "${YAW0}" --elev "${ELEV}" --turns "${TURNS:-0}" \
  --n-frames "${NFR}" --res "${RES:-518}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
