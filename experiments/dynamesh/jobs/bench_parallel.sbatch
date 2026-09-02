#!/bin/bash
# bench_parallel.sbatch — sequential vs frame-batched inference, same adapter,
# same mesh, same noise. Zero-shot transfer: hand_rorschach adapter -> alien.
#
# Times INFERENCE ONLY (12-step ODE + decode). Rasterisation is excluded: a
# frame's PBR field is generated once and each view is a grid_sample from it.
#
# Frame batching is valid because every frame shares the same noise and the same
# sparse structure S, and no frame reads another frame's output.
#SBATCH --job-name=benchpar
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/benchpar_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/benchpar_%x_%j.log
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
RUN=${RUN:-${E}/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_eaa1fb67}
MESH=${MESH:-${T2}/data/alien_glow/mesh/alien_glow_render_frame.obj}
COND=${COND:-${T2}/data/hand_rorschach/frames_from_video}
# VIEWS arrives colon-separated: sbatch --export splits its own list on commas,
# so VIEWS=0,90,180,270 silently reached the script as just "0" and the first
# smoke run rendered one view instead of four.
NF=${NF:-150}; VIEWS=${VIEWS:-0:90:180:270}; VIEWS=${VIEWS//:/,}
# MCFM is the blend mode (v2_D=W3, E=W5, F=W7, G=W11, H=W13, I=W15). Leave it EMPTY
# for the W=1 arm: that adapter was trained without MCFM, and passing the flag anyway
# would trip the trainer's own GATE-mcfm train/test-mismatch check.
MCFM=${MCFM-v2_D}
TAGBASE=${TAGBASE:-bench_alien}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} NF=${NF} BATCHES='${BATCHES}' ==="
nvidia-smi -L
for B in ${BATCHES:-1 8}; do
  echo "--- batch=${B} ---"
  ${PY2} -u "${E}/bench_transfer_parallel.py" \
    --run "${RUN}" --mesh "${MESH}" --cond-dir "${COND}" \
    ${MCFM:+--mcfm ${MCFM}} \
    --sweep both --turns 0 --n-frames "${NF}" --res 518 \
    --views "${VIEWS}" --batch "${B}" \
    ${SHARD_I:+--shard-idx ${SHARD_I}} ${SHARD_N:+--shard-n ${SHARD_N}} \
    --tag "${TAGBASE}_b${B}" \
    --timing-json "${E}/out/timing_${TAGBASE}_b${B}.json" 2>&1 \
    | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
done
echo "=== Done $(date) ==="
