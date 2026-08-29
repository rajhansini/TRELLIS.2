#!/bin/bash
# rung37: cross-attn LoRA + self-attn LoRA + a THIRD temporal LoRA whose attention
# is restricted to the SAME SPATIAL POSITION across the window (MCFM v2 geometry,
# learned). Stamps rung:37 and temporal_geometry:per_position, so it can never
# share a directory or a checkpoint with rung31 (joint spatio-temporal).
# Pass OBJ NFR MESH GTDIR. SMOKE=1 -> 2 epochs, 8 frames.
#SBATCH --job-name=r37
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# 04:00:00 -> 08:00:00. 21 of the first 42 jobs hit TIMEOUT at exactly 4h and the
# runs that DID finish took 2:54-3:59, so 4h was never headroom -- rung37 runs the
# cross-attention twice per block plus the per-position mixer, ~3-4x an m3 arm.
#SBATCH --time=04:00:00   # cluster MAX: every partition caps at 4:00:00, so 8h
                          # is rejected outright ("Requested time limit is invalid").
                          # A 30-epoch run needs ~3-6h, so some cells WILL hit the
                          # wall. That is fine: the trainer resumes from
                          # ckpts/lora_e*.pt, and watchdog.py resubmits r37_ on
                          # TIMEOUT, so each 4h slice adds epochs until 30 are done.
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r37_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r37_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
: "${OBJ:?}"; : "${NFR:?}"; : "${MESH:?}"; : "${GTDIR:?}"
cd ${T2}; echo "=== $(date) host=$(hostname) OBJ=${OBJ} nfr=${NFR} SMOKE=${SMOKE:-0} ==="; nvidia-smi -L
${PY2} -u "${E}/rung37_perpos_temporal_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --temporal-window 3 --branch both --gate-init 0.0 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"
echo "=== Done $(date) ==="
