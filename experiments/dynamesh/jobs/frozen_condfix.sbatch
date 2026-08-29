#!/bin/bash
# frozen_condfix.sbatch — a TRUE frozen TRELLIS.2 baseline render.
#
# Fixes two things that made the frozen panel in the limitations figure wrong:
#
#  1. CONDITIONING MATTE.  render_rung27_orbit.py builds the conditioning alpha
#     with a hardcoded r.min(axis=2) < 245.  That assumes a near-white backdrop.
#     penguin_circuits was shot on a GREY backdrop measuring 237, so 99.7% of
#     every frame passes the test, the union crop degenerates to (0,0,958,958)
#     and DINOv3 receives a raw 960x960 frame.  The file's own comment says that
#     input "yields a black texture" — which is exactly the black penguin at
#     9.58 dB.  A corrected threshold is still not enough: it picks up the
#     contact shadow and widens the crop.  --cond-mask uses the MESH render
#     mask, which is exact, shadow-free and identical on every frame.
#
#  2. MCFM LEAKAGE.  decode_both() deliberately feeds BOTH arms the same cond,
#     blended included, to isolate the adapter.  Rendered from an --mcfm run the
#     "frozen" arm is therefore frozen + MCFM, not the paper's baseline.  RUN
#     here points at the NON-mcfm penguin run, so the tokens are vanilla.
#
# --frozen-only skips the adapter pass entirely: one panel, no LoRA.
#SBATCH --job-name=frzfix
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frzfix_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frzfix_%x_%j.log
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
: "${RUN:?}"; : "${MASK:?}"; : "${YAW:?}"; : "${TAG:?}"; : "${NFR:?}"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} YAW=${YAW} TAG=${TAG} NFR=${NFR} ==="
nvidia-smi -L
${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" \
  --sweep both --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --yaw0 "${YAW}" \
  --res "${RES:-518}" --frozen-only --cond-mask "${MASK}" --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
N=$(ls "${E}/out/${TAG}/frames"/*.png 2>/dev/null | wc -l)
[ "$N" -eq "${NFR}" ] || { echo "FAILED: ${N} frames written, expected ${NFR}"; exit 1; }
echo "wrote ${N} frames to ${E}/out/${TAG}/frames"
echo "=== Done $(date) ==="
