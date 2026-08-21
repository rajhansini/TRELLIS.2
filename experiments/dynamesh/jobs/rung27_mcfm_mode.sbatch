#!/bin/bash
# rung27 + MCFM v2_D on a hero-posed object (pose known by construction, no orientation search).
# v2_D chosen because it won on BOTH already-trained objects and the ordering was
# identical on each: spot_lava v2_D 24.194 > v2_C 24.061 >> v3_C 22.668 ~ v3_D 22.659;
# teapot_lava2 v2_D 22.688 > v2_C 22.675 >> v3_D 20.336 ~ v3_C 20.131.
# MCFM blends the cached DINOv3 conditioning temporally before training; it needs
# no per-object precompute and derives its window from conds.keys(), so 121 frames
# behaves exactly like 150.
# One object per job. Pass OBJ and NFR.
#   --mesh           data/<obj>/mesh/<obj>_render_frame.obj   pose baked in
#   --gt-dir         data/<obj>/frames_from_video                  conditioning
#   --gt-render-dir  out/gt_targets_<obj>/frames              the 2D copy
# GATE-align asserts our training render and the target mask are the SAME pixels,
# so a mesh/target mismatch cannot slip through silently.
#SBATCH --job-name=r27m3
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27m3_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27m3_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}; echo "=== $(date) host=$(hostname) OBJ=${OBJ} nfr=${NFR} MODE=${MODE} MESH=${MESH} GTDIR=${GTDIR} ==="; nvidia-smi -L

${PY2} -u "${E}/rung27_selfattn_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 --mcfm "${MODE}" \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"
echo "=== Done $(date) ==="
