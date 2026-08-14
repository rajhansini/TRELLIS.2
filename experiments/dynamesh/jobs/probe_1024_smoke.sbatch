#!/bin/bash
# rung20 on teapot_lava2 — the re-generated lava video, white background, effect
# confined to the teapot's surface. Same mesh and camera as the original teapot.
#
# WHY THIS DATASET: the original teapot video grew its object 102,549 -> 109,583 px
# (6.9%) and left 319 px of our silhouette uncovered, which step 3 of pipeline B
# had to flood-fill. teapot_lava2 drifts 1.1% and covers our silhouette in ALL
# 150 frames (0 px uncovered, worst case), so every supervised pixel carries real
# video colour and the flood-fill never activates. Gates run before submission:
#   [GATE-bg]  corner RGB (252.1, 253.1, 252.6)              PASS
#   [GATE-sil] silhouette drift 1.1% over the clip           PASS
#   [CONTAIN]  0 px of our silhouette uncovered, all frames  PASS
#   frame 1 vs mesh render: IoU 0.9814, centroid off (0.2, 0.4) px
#
# Targets are the 2D copy (option B). Backprojection is NOT used and is not
# needed: its only advantage is attaching colour to a surface point, which buys
# nothing at a fixed camera, and it leaves a ~1% unfilled fringe.
#SBATCH --job-name=r20_1024smoke
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 4 h is the cap on every partition here. The run checkpoints every epoch and
# --requeue resumes from the latest, guarded by GATE-resume on mesh/targets/res.
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r20_1024smoke_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r20_1024smoke_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
MESH=/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply
GTDIR=${T2}/data/teapot_lava2/frames_from_video    # 150 frames @960, white bg
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
echo "MESH  ${MESH}"
echo "GTDIR ${GTDIR}"

# ---- 1. silhouette at 960^2 --------------------------------------------------
# --raw is REQUIRED: frozen_f0075.ply is already in the render frame and the
# training loop rasterises v_raw. Without it the mask comes out 136,643 px
# instead of 98,083 and GATE-align aborts the run (this happened on job 2173382).
# Own tag rather than sharing teapot_960_mask, so this cannot race job 2173413
# reading a half-written .npy.
if [ ! -f "${E}/out/teapot_lava2_mask/render_mask.npy" ]; then
  echo "=== [1/3] render mask @960 (--raw) ==="
  ${PY2} -u "${E}/make_render_mask.py" \
    --mesh "${MESH}" --res 960 --raw --tag teapot_lava2_mask
else
  echo "=== [1/3] render mask already present, skipping ==="
fi

# ---- 2. 2D-copy targets (option B) at 960^2 ----------------------------------
if [ ! -f "${E}/out/gt_targets_teapot_lava2/frames/valid_0150.npy" ]; then
  echo "=== [2/3] 2D-copy targets @960 ==="
  ${PY2} -u "${E}/make_gt_targets.py" \
    --gt-dir "${GTDIR}" \
    --render-mask "${E}/out/teapot_lava2_mask/render_mask.npy" \
    --render-res 960 --n-frames 150 \
    --tag gt_targets_teapot_lava2
else
  echo "=== [2/3] targets already present, skipping ==="
fi

# ---- 3. rung20 ---------------------------------------------------------------
# Identical hyperparameters to the original teapot run and both spot runs.
# render_res is in the run-id hash now, so this cannot collide with bbbce460.
echo "=== [3/3] rung20 train @960 ==="
${PY2} -u "${E}/rung20_conf_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${GTDIR}" \
  --gt-render-dir "${E}/out/gt_targets_teapot_lava2/frames" \
  --render-res 960 --resolution 1024 \
  --targets kv --blocks all --rank 4 \
  --conf-weight --conf-tau 0.25 --conf-quantile -1 \
  --epochs 2 --n-frames 8 --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
