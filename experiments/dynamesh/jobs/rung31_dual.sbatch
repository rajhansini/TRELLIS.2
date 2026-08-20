#!/bin/bash
# rung31 — LoRA on the TEMPORAL axis of cross-attention, alongside the spatial one.
#
# Cloned from jobs/rung27_spot_lava.sbatch. Every training flag is byte-identical
# to the rung27 baseline of the same object, so the ONLY difference is the two
# new flags. That is what makes the comparison a single-variable one:
#     --temporal-window 3   context becomes [f-1, f, f+1], 1029 -> 3087 tokens
#     --branch both         one LoRA set per axis
#
# BASELINES this is measured against (rung27, no MCFM, 30 ep, r4, s42, res 960):
#     spot_lava 23.994   skull_lava 25.347   pumpkin_rot 33.723   hand_rorschach 24.831
#
# a30 is NOT in the constraint: 14 renders died on c001 with 'CUDA error: no
# kernel image is available for execution'. a40|L40S only.
#
# Usage:  sbatch --export=ALL,OBJ=spot_lava           jobs/rung31_dual.sbatch
# Smoke:  sbatch --export=ALL,OBJ=spot_lava,SMOKE=1 --time=00:40:00 jobs/rung31_dual.sbatch
#SBATCH --job-name=r31
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 4 h partition cap. rung31 runs ~2.5-3x rung27 (two cross-attn passes, 3x the
# context), so a 150-frame run WILL exceed one slot. --requeue plus the script's
# per-epoch checkpoint and GATE-resume carries it across as many slots as needed.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r31_%x_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true

: "${OBJ:?set OBJ=spot_lava|skull_lava|pumpkin_rot|hand_rorschach}"
case "$OBJ" in
  spot_lava)      MESH=spot_render_frame.obj;              GT=gt_targets_spot_lava;        NF=150 ;;
  skull_lava)     MESH=skull_lava_render_frame.obj;        GT=gt_targets_skull_lava;       NF=150 ;;
  pumpkin_rot)    MESH=pumpkin_rot_render_frame_guan.obj;  GT=gt_targets_pumpkin_rot_guan; NF=121 ;;
  hand_rorschach) MESH=hand_rorschach_render_frame.obj;    GT=gt_targets_hand_rorschach;   NF=150 ;;
  *) echo "unknown OBJ=$OBJ" >&2; exit 2 ;;
esac
MESH_P="${T2}/data/${OBJ}/mesh/${MESH}"
GT_P="${T2}/data/${OBJ}/frames_from_video"
GTR_P="${E}/out/${GT}/frames"
for p in "$MESH_P" "$GT_P" "$GTR_P"; do
  [ -e "$p" ] || { echo "MISSING INPUT: $p" >&2; exit 3; }
done

cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} obj=${OBJ} nf=${NF} smoke=${SMOKE:-0} ==="
nvidia-smi -L

${PY2} -u "${E}/rung31_dual_attn_lora.py" \
  --mesh "${MESH_P}" --gt-dir "${GT_P}" --gt-render-dir "${GTR_P}" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames ${NF} --seed 42 --lr 1e-4 \
  --temporal-window 3 --branch "${BRANCH:-both}" --gate-init 0.0 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
