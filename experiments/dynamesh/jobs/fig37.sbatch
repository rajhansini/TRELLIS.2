#!/bin/bash
# fig37: rung37_perpos_temporal_lora.py, window as a PARAMETER (TW, default 3).
# Mirrors fig31.sbatch so the w11 arm submits the same way rung31's did. Every
# path is a parameter: hardcoding hero paths would train the guan-pipeline objects
# against the wrong targets at the wrong frame count and still exit 0.
#SBATCH --job-name=fig37
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G      # same as fig31, which carries the HEAVIER W=11 context
#                      # (attends 11,319 keys); rung37 collapses to 1029 first.
#SBATCH --time=04:00:00   # partition MaxTime is 04:00:00 on ALL partitions -- a longer
                          # request is REJECTED at submit. rung37 runs cross-attn twice
                          # per block plus the mixer, so TIMEOUT is expected; the
                          # trainer resumes from ckpts/lora_e*.pt and watchdog.py
                          # resubmits on TIMEOUT until 30 epochs are done.
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
mkdir -p ${E}/out/FIGRUNS
: "${OBJ:?}"; : "${NFR:?}"; : "${MESH:?}"; : "${GTDIR:?}"
cd ${T2}
echo "=== $(date) host=$(hostname) OBJ=${OBJ} ARM=fig37 TW=${TW:-3} MODE=${MODE:-none} nfr=${NFR} ==="
echo "    MESH=${MESH}"
echo "    GTDIR=${GTDIR}"
nvidia-smi -L
${PY2} -u "${E}/rung37_perpos_temporal_lora.py" \
  --mesh "${MESH}" --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
  --temporal-window ${TW:-3} --branch both --gate-init 0.0 ${MODE:+--mcfm ${MODE}} 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
