#!/bin/bash
# Full MCFM x rung grid. Flags match the completed baselines EXACTLY except --mcfm,
# so each run has a direct unblended counterpart already on disk:
#   rung17=kv  rung18=qkv  rung19=qkvo  rung25=kvo,  all --recon l1 --lpips
#   --render-res 960 --resolution 512 --epochs 30 --n-frames 150 --seed 42 --lr 1e-4
# Meshes are taken from the baseline jobs verbatim: spot_lava uses the FITTED
# spot_render_frame.obj, teapot_lava2 uses frozen_f0075.ply (NOT the _uv.obj —
# that was only used for MCFM inference probes and would break comparability).
#
#   MCFM=<v2_C|v2_D|v3_C|v3_D>  TARGETS=<kv|qkv|qkvo|kvo>  ASSET=<spot_lava|teapot_lava2>
#
# GPU IS PINNED TO L40S: the trellis2 env's CUDA extensions have kernel images for
# sm86 (A40) and sm89 (L40S) only. A bare --gres=gpu:1 lands on A30/rtx2080ti and
# dies with "no kernel image is available" or nvdiffrast CUDA 209.
#SBATCH --job-name=mcfmfull
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mcfmfull_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mcfmfull_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ASSET=${ASSET} TARGETS=${TARGETS} MCFM=${MCFM} ==="
nvidia-smi -L

case "$ASSET" in
  spot_lava)    MESH="${T2}/data/spot_lava/mesh/spot_render_frame.obj" ;;
  teapot_lava2) MESH="/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply" ;;
  *) echo "unknown ASSET=$ASSET"; exit 1 ;;
esac

${PY2} -u "${E}/rung26_mcfm_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${ASSET}/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_${ASSET}/frames" \
  --render-res 960 --resolution 512 \
  --targets ${TARGETS} --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  --mcfm ${MCFM} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
