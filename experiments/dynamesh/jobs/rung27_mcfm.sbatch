#!/bin/bash
# rung27 (qkvo+sa) + MCFM temporal token blending.
#
# Mirrors rung27_{spot_lava,teapot_lava2}.sbatch EXACTLY and adds --mcfm alone,
# so every run has a direct unblended control already on disk:
#     spot_lava     23.994 / 0.8317      teapot_lava2  22.624 / 0.7570
#
#   MCFM=<v2_C|v2_D|v3_C|v3_D>   ASSET=<spot_lava|teapot_lava2>
#
# GPU: A40 OR L40S, not L40S alone. The trellis2 env's compiled CUDA extensions
# carry kernel images for sm86 and sm89 ONLY, so a bare --gres=gpu:1 can land on an
# A30 or rtx2080ti and die instantly with "no kernel image is available for
# execution" — that hazard is real and this constraint still excludes it. But sm86
# IS the A40, so pinning gpu:L40S:1 excluded the A40 too, and with it 14 of the 21
# eligible nodes. Every one of these jobs then sat on (Priority) behind other
# users' L40S work while A40 cards were free. Measured peak is 19.8 GiB against
# 48 GiB on both cards, so the two are interchangeable here.
#
# A30 stays out deliberately: it is sm80 (no kernel image) and only 24 GiB, which
# is 82% utilisation at our measured peak.
#SBATCH --job-name=r27mcfm
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27mcfm_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27mcfm_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ASSET=${ASSET} MCFM=${MCFM} ==="
nvidia-smi -L

case "$ASSET" in
  spot_lava)    MESH="${T2}/data/spot_lava/mesh/spot_render_frame.obj" ;;
  teapot_lava2) MESH="/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply" ;;
  *) echo "unknown ASSET=$ASSET"; exit 1 ;;
esac

${PY2} -u "${E}/rung27_selfattn_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${ASSET}/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_${ASSET}/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  --mcfm ${MCFM} \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
