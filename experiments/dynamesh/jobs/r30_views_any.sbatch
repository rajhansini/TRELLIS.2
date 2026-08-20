#!/bin/bash
# r30_fixedviews.sbatch — ONE fixed camera, 150 frames, for the horse transfer.
#
# WHY THIS AND NOT ANOTHER ORBIT. In a rotating video a texture change and a
# viewpoint change look identical, so an orbit cannot answer "is the texture ON
# the surface, or painted at the training angle". Four FIXED cameras can:
#   yaw0=0    the training view      the adapter was supervised here
#   yaw0=90   unseen                 never supervised
#   yaw0=180  unseen, opposite side  the hardest case
#   yaw0=270  unseen
# Camera held still, frame advances 1..150, so the only thing moving in each
# video is the texture. Compare the four side by side and a texture that only
# exists at yaw 0 becomes obvious immediately.
#
# --turns 0 keeps the azimuth constant; --yaw0 sets which azimuth.
# usage: OBJ=pumpkin_rot YAW=90 TAG=r30_pumpkin_yaw090 sbatch jobs/r30_fixedviews.sbatch
# OBJ selects the TARGET mesh (the one the adapter never saw). The adapter and
# its conditioning stay spot_lava in every case, so the only thing that varies
# across objects is the geometry -- which is the whole point of the transfer.
#SBATCH --job-name=r30fixed
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r30fixed_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r30fixed_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} OBJ=${OBJ:-horse_metal} YAW=${YAW} TAG=${TAG} ==="; nvidia-smi -L
${PY2} -u "${E}/render_rung30_transfer.py" \
  --run "${E}/runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94" \
  --mesh "${T2}/data/${OBJ:-horse_metal}/mesh/${OBJ:-horse_metal}_render_frame_guan.obj" \
  --cond-dir "${T2}/data/spot_lava/frames_from_video" \
  --sweep both --n-frames 150 --turns "${TURNS:-0}" --yaw0 "${YAW:-0}" --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
echo "=== Done $(date) ==="
