#!/bin/bash
# ONE (object, arm, view) per job — 64 of these are independent and must never be
# looped inside a single job. Pass RUN, TAG, NFR, YAW0, ARM. ELEV/TURNS/RES have
# defaults matching the comparison panel (fixed camera, 518 px).
#
# WHY ONE TEMPLATE FOR ALL FOUR ARMS. rung31 needs render_rung31_orbit.py (its
# checkpoint has 510 tensors: doubled cross-attn for branch=both plus 30 per-block
# gates, so the 300-tensor rung27 registry raises "Unexpected key(s)"). The other
# three go through render_rung27_orbit.py. Driving those from two separate sbatch
# files let them drift -- jobs/rung31_render.sbatch hardcodes --res 720 and omits
# --n-angles, which would have put a differently-sized, differently-sampled tile
# in the same row as the others. Dispatching here keeps every tile identical
# except the adapter, which is the entire point of the panel.
#
# TURNS=0 pins the camera at (YAW0, ELEV) for the whole sequence, so all change on
# screen is texture, not viewpoint.
#SBATCH --job-name=pview
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"   # nvdiffrast has no kernel for h100/h200 (CUDA 209)
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/pview_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/pview_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
: "${RUN:?set RUN}"; : "${TAG:?set TAG}"; : "${NFR:?set NFR}"; : "${ARM:?set ARM}"
# RUN may be given relative to dynamesh/ (e.g. "runs/rung27_..."), but this script
# cd's to TRELLIS.2 so the renderer resolves it against the WRONG directory and
# dies on a missing config.json. Absolutize it here rather than at every call site.
case "$RUN" in /*) ;; *) RUN="${E}/${RUN}";; esac
[ -f "${RUN}/config.json" ] || { echo "no config.json under RUN=${RUN}"; exit 1; }
YAW0=${YAW0:-0}; ELEV=${ELEV:-15}; TURNS=${TURNS:-0}; RES=${RES:-518}
echo "=== $(date) host=$(hostname) ARM=${ARM} RUN=${RUN} TAG=${TAG} yaw0=${YAW0} ==="
nvidia-smi -L

# Idempotent: a requeued job must not redo finished work. Do NOT use
# `ls glob | wc -l` -- under `set -eo pipefail` a missing dir makes ls exit 2,
# pipefail propagates it and set -e kills the job, which silently broke every
# FIRST render of a new object.
HAVE=0
if [ -d "${E}/out/${TAG}/frames" ]; then
  HAVE=$(find "${E}/out/${TAG}/frames" -maxdepth 1 -name "*.png" | wc -l)
fi
if [ "${HAVE}" -ge "${NFR}" ]; then
  echo "ALREADY COMPLETE: ${TAG} has ${HAVE}/${NFR} frames"; exit 0
fi
echo "have ${HAVE}/${NFR} frames, rendering"

case "${ARM}" in
  rung31) SCRIPT=render_rung31_orbit.py ;;
  *)      SCRIPT=render_rung27_orbit.py ;;
esac
echo "dispatching to ${SCRIPT}"

${PY2} -u "${E}/${SCRIPT}" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" \
  --turns "${TURNS}" --yaw0 "${YAW0}" --elev "${ELEV}" \
  --res "${RES}" --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
