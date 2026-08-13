#!/bin/bash
# rung20 on the teapot, re-run at 960^2 so its row is comparable with spot's.
#
# WHY THIS EXISTS: the original teapot rung20 (bbbce460) trained at --render-res
# 518 while both spot runs trained at 960. PSNR is scored over the supervised
# silhouette, which is 28,559 px at 518 and ~98,000 px at 960 for this mesh, so
# the teapot and spot rows were never like-for-like. This run removes that.
#
# The 2D-copy targets (option B) are the ones used by every rung; the
# backprojected set is NOT used. Targets at 960 do not exist yet, so steps 1-2
# build them before training. Steps 1-2 are cheap; step 3 is the 30-epoch run.
#SBATCH --job-name=r20_teapot960
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 4 h is the cap on every partition here. The run checkpoints every epoch and
# --requeue resumes from the latest, guarded by GATE-resume on mesh/targets/res.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r20teapot960_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r20teapot960_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
MESH=/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply
GTDIR=${T2}/data/teapot/frames_from_video          # 150 frames, white bg (252.9)
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
echo "MESH  ${MESH}"
echo "GTDIR ${GTDIR}"

# ---- 1. rasterise the teapot silhouette at 960^2 through the pipeline camera --
if [ ! -f "${E}/out/teapot_960_mask/render_mask.npy" ]; then
  echo "=== [1/3] render mask @960 ==="
  # --raw is REQUIRED here. frozen_f0075.ply is already in the render frame
  # (export_frame_mesh.py --alignment), and the training loop rasterises v_raw.
  # Without --raw the script re-centres/re-scales/axis-swaps the mesh and the
  # mask comes out 136,643 px instead of 98,083 — GATE-align catches it, but
  # only after the targets have been built from the wrong silhouette.
  ${PY2} -u "${E}/make_render_mask.py" \
    --mesh "${MESH}" --res 960 --raw --tag teapot_960_mask
else
  echo "=== [1/3] render mask @960 already present, skipping ==="
fi

# ---- 2. 2D-copy targets (option B) at 960^2 ---------------------------------
if [ ! -f "${E}/out/gt_targets_teapot_960/frames/valid_0150.npy" ]; then
  echo "=== [2/3] 2D-copy targets @960 ==="
  ${PY2} -u "${E}/make_gt_targets.py" \
    --gt-dir "${GTDIR}" \
    --render-mask "${E}/out/teapot_960_mask/render_mask.npy" \
    --render-res 960 --n-frames 150 \
    --tag gt_targets_teapot_960
else
  echo "=== [2/3] targets @960 already present, skipping ==="
fi

# ---- 3. rung20 ---------------------------------------------------------------
# Every hyperparameter below is identical to the original teapot run and to both
# spot runs. --render-res is the single deliberate change, and it is now part of
# the run-id hash, so this cannot land in bbbce460's directory.
echo "=== [3/3] rung20 train @960 ==="
${PY2} -u "${E}/rung20_conf_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${GTDIR}" \
  --gt-render-dir "${E}/out/gt_targets_teapot_960/frames" \
  --render-res 960 --resolution 512 \
  --targets kv --blocks all --rank 4 \
  --conf-weight --conf-tau 0.25 --conf-quantile -1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
