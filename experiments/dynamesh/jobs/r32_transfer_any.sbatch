#!/bin/bash
# r32_transfer_any.sbatch — zero-shot transfer, fully parameterised, NO TRAINING.
#
# Runs render_rung31_transfer_mcfm.py, which contains no optimizer, no backward
# and no .step(); GATE-notrain asserts it at runtime. The adapter is loaded from
# an ALREADY-TRAINED lora_best.pt and is read-only for the whole run.
#
# Both arms go through this ONE script so the code path is identical and the only
# differences are the checkpoint and whether the conditioning is blended:
#   vanilla  RUN=<mcfm=None run>   MCFM=""      -> decode_both re-encodes per frame
#   MCFM     RUN=<mcfm=v2_D run>   MCFM=v2_D    -> all frames encoded, then blended
# GATE-mcfm refuses to run if the checkpoint's training mode and MCFM disagree,
# so the two can never be crossed by accident.
#
# usage: RUN=<abs run dir> MESH=<abs .obj> COND=<abs frames dir> MCFM=v2_D \
#        YAW=90 TURNS=0 NF=150 TAG=r32_mcfm_skull_yaw090 sbatch jobs/r32_transfer_any.sbatch
#SBATCH --job-name=r32xfer
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r32xfer_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r32xfer_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} TAG=${TAG} ==="
echo "    RUN=${RUN}"; echo "    MESH=${MESH}"; echo "    COND=${COND}  MCFM=${MCFM:-none}  YAW=${YAW:-0}  TURNS=${TURNS:-0}"
nvidia-smi -L
${PY2} -u "${E}/render_rung31_transfer_mcfm.py" \
  --run "${RUN}" --mesh "${MESH}" --cond-dir "${COND}" \
  ${MCFM:+--mcfm ${MCFM}} \
  --sweep both --n-frames "${NF:-150}" --turns "${TURNS:-0}" --yaw0 "${YAW:-0}" \
  --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
