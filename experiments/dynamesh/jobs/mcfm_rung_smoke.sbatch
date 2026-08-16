#!/bin/bash
# Smoke for the --mcfm flag on rung20_conf_lora.py.
#   MCFM=<v2_C|v2_D|v3_C|v3_D>   omit for the VANILLA control
#   TARGETS=<kv|qkv|qkvo|kvo>    ASSET=<spot_lava|teapot_lava2>
# 2 epochs, 8 frames. What it must show:
#   vanilla  -> no [MCFM] lines, and a RUN_ID matching the existing baseline runs
#   mcfm     -> [MCFM] GATE-blend > 0, and a DIFFERENT RUN_ID per mode
#SBATCH --job-name=mcfmsmoke
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mcfmsmoke_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mcfmsmoke_%x_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} MCFM=${MCFM:-VANILLA} ==="
nvidia-smi -L

ASSET=${ASSET:-spot_lava}
TARGETS=${TARGETS:-kv}
case "$ASSET" in
  spot_lava)    MESH="${T2}/data/spot_lava/mesh/spot_render_frame.obj" ;;
  teapot_lava2) MESH="/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075_uv.obj" ;;
  *) echo "unknown ASSET=$ASSET"; exit 1 ;;
esac

${PY2} -u "${E}/rung20_conf_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${ASSET}/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_${ASSET}/frames" \
  --render-res 960 --resolution 512 \
  --targets ${TARGETS} --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 --smoke \
  ${MCFM:+--mcfm ${MCFM}} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
