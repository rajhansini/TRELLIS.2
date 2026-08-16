#!/bin/bash
# Renders for every completed teapot_lava2 KL arm: single view, 360, 720.
#SBATCH --job-name=klrender_t2
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/klrender_t2_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/klrender_t2_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L
F="${E}/kl_teapot_runs.tsv"
NOISE='FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward|it/s\]'
while IFS=$'\t' read -r W R; do
  echo ""; echo "######## teapot_lava2  w_kl=${W}  ${R} ########"
  echo "---- single view, 150 frames ----"
  ${PY2} -u "${E}/render_rung14_t2_video.py" --run "${E}/runs/${R}" --ckpt lora_e030.pt \
    --n-frames 150 --tag "kl${W}_single" 2>&1 | stdbuf -oL grep --line-buffered -vE "$NOISE" | tail -4
  echo "---- 360, 150 angles ----"
  ${PY2} -u "${E}/render_rung17_orbit.py" --run "${E}/runs/${R}" --ckpt lora_e030.pt \
    --sweep angle --n-angles 150 --turns 1 --res 720 --tag "kl${W}_360" \
    2>&1 | stdbuf -oL grep --line-buffered -vE "$NOISE" | tail -4
  echo "---- 720 (two turns), 150 angles ----"
  ${PY2} -u "${E}/render_rung17_orbit.py" --run "${E}/runs/${R}" --ckpt lora_e030.pt \
    --sweep angle --n-angles 150 --turns 2 --res 720 --tag "kl${W}_720" \
    2>&1 | stdbuf -oL grep --line-buffered -vE "$NOISE" | tail -4
done < "$F"
echo "=== Done $(date) ==="
