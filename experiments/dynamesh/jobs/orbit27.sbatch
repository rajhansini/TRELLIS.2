#!/bin/bash
# rung27 turntable. Same shape as orbit3.sbatch but calls render_rung27_orbit.py,
# which imports rung27_selfattn_lora — the only trainer whose CrossAttnLoRA has
# sa_qkv / sa_out. Through render_rung17_orbit.py a rung27 checkpoint raises
# "Unexpected key(s) in state_dict".
#
#   RUN=<run dir>  TAG=<output tag>  TURNS=<0|1|2>
#     TURNS=0  side   camera FIXED at the training view, frames 1..150
#     TURNS=1  360    one revolution across the sequence
#     TURNS=2  720    two revolutions — every surface point seen TWICE at
#                     different points in the texture evolution
#SBATCH --job-name=orbit27
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"   # nvdiffrast has no kernel for h100/h200 (CUDA 209 on h002)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orbit27_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orbit27_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) RUN=${RUN} TAG=${TAG} TURNS=${TURNS:-1} ==="; nvidia-smi -L

${PY2} -u "${E}/render_rung27_orbit.py" --run "${RUN}" --sweep both \
  --n-frames "${NFR:-150}" --n-angles "${NFR:-150}" --turns "${TURNS:-1}" --tag "${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
