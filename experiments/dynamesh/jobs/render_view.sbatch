#!/bin/bash
# One (object, arm, view) per job — 60 of these are independent and must never be
# looped inside one job.  Pass RUN, TAG, NFR, YAW0, ELEV, TURNS.
#
#   TURNS=0  camera FIXED for the whole sequence at (YAW0, ELEV) — all change is
#            texture, which is what the diagonal and training views are for
#   TURNS=1  one revolution across the sequence  (360)
#   TURNS=2  two revolutions — every surface point seen TWICE at different points
#            in the texture evolution  (720)
#
# render_rung27_orbit.py imports rung27_selfattn_lora, the only trainer whose
# CrossAttnLoRA carries sa_qkv/sa_out; through render_rung17_orbit.py a rung27
# checkpoint raises "Unexpected key(s) in state_dict".
#SBATCH --job-name=rview
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"   # nvdiffrast has no kernel for h100/h200 (CUDA 209)
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G   # measured peak RSS 9.2G over 36 completed renders (mean 7.0),
#                    so 24G is 2.6x headroom. This is inference: no optimiser state,
#                    no gradient buffers. 64G cost 26 of the group's 1024 billing
#                    units and capped concurrency at 39; 24G costs 16 and lifts it to 64.
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/rview_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/rview_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) RUN=${RUN} TAG=${TAG} yaw0=${YAW0} elev=${ELEV} turns=${TURNS} ==="
nvidia-smi -L

# Idempotent: a requeued job must not redo 20 minutes of finished work.
# NOTE: must not use `ls glob | wc -l` here. Under `set -eo pipefail`, a missing
# frames dir makes ls exit 2, pipefail propagates it, and set -e kills the job --
# which silently broke every FIRST render of a new object.
HAVE=0
if [ -d "${E}/out/${TAG}/frames" ]; then
  HAVE=$(find "${E}/out/${TAG}/frames" -maxdepth 1 -name "*.png" | wc -l)
fi
if [ "${HAVE}" -ge "${NFR}" ]; then
  echo "ALREADY COMPLETE: ${TAG} has ${HAVE}/${NFR} frames — nothing to do"; exit 0
fi
echo "have ${HAVE}/${NFR} frames, rendering"

${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" \
  --turns "${TURNS}" --yaw0 "${YAW0}" --elev "${ELEV}" \
  --res "${RES:-518}" --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
