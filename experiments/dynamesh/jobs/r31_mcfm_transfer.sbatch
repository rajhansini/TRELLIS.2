#!/bin/bash
# r31_mcfm_transfer.sbatch — the MCFM arm of the zero-shot transfer.
#
# Matched pair with jobs/r30_fixedviews.sbatch. EVERYTHING is held identical
# except two things, so any difference in the output is attributable to MCFM:
#   --run    the LoRA trained on spot_lava WITH mcfm=v2_D, instead of the
#            vanilla one (rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94)
#   --mcfm   v2_D blending on the conditioning at render time, which is what
#            that adapter was fitted against
# Same target meshes, same --cond-dir (spot_lava's own frames), same 150
# frames, same camera code. GATE-mcfm in the renderer refuses the run if the
# checkpoint's training mode and --mcfm disagree.
#
# usage: OBJ=horse_metal YAW=90 TURNS=0 TAG=r31_horse_yaw090 sbatch jobs/r31_mcfm_transfer.sbatch
#   TURNS 0 = camera fixed at azimuth YAW, frames 1..150 (the four-view test)
#   TURNS 1 = one revolution across the sequence;  TURNS 2 = two revolutions
#SBATCH --job-name=r31mcfm
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31mcfm_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31mcfm_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} OBJ=${OBJ} YAW=${YAW:-0} TURNS=${TURNS:-0} TAG=${TAG} ==="
nvidia-smi -L
${PY2} -u "${E}/render_rung31_transfer_mcfm.py" \
  --run "${E}/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_59c51f6e" \
  --mesh "${T2}/data/${OBJ}/mesh/${OBJ}_render_frame_guan.obj" \
  --cond-dir "${T2}/data/spot_lava/frames_from_video" \
  --mcfm v2_D \
  --sweep both --n-frames 150 --turns "${TURNS:-0}" --yaw0 "${YAW:-0}" \
  --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
