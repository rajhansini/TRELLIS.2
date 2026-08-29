#!/bin/bash
# rung26_mcfm_lora.py with --targets qkvo -> stamps rung: 19, mcfm: <MODE>.
# This is the "temporal + cross-attention" cell of the REVERSED ablation ladder
# (frozen -> +temporal -> +temporal+CA -> +temporal+CA+SA), the ordering a reader
# asks for when the temporal component is introduced last in the forward ladder.
#
# WHY rung26 AND NOT rung19. There is no rung19 script. _TARGET_RUNG in
# rung26_mcfm_lora.py maps targets->rung {'kv':17,'qkv':18,'qkvo':19,'kvo':25,'v':26},
# so --targets qkvo IS rung 19; the filename's "26" is a file index, not the rung it
# produces. rung26_mcfm_lora.py is rung20_conf_lora.py plus five --mcfm-gated lines
# (verified by diff), and its own --mcfm help states that with --mcfm unset the two
# scripts are equivalent.
#
# FLAGS THAT MUST BE EXPLICIT -- the defaults do NOT reproduce rung19:
#   --targets qkvo   default is qkv  (that would be rung 18)
#   --recon l1       default is l2
#   --lpips          store_true, off by default
#   --w-lpips 0.1    inert without --lpips
# Everything else matches the published rung19 config: rank 4, alpha 4, lr 1e-4,
# 30 epochs, seed 42, blocks all, conf_weight False, loss_scale 4096.
#
# ROW 2 COMES FREE. render/decode runs the frozen and adapted arms from the same
# noise AND the same blended conditioning ("the blend is a property of the input, not
# of the adapter"), so each texel json also carries frozen_* -- i.e. frozen+MCFM,
# which completes the reversed ladder's second row at no extra cost.
#SBATCH --job-name=r26ca
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# a40|L40S only: the trellis2 env dies with GLIBC_2.32 importing o_voxel._C on a30.
# Must live in the SCRIPT -- watchdog.py resubmits with only --job-name and --export,
# so a CLI --constraint is silently dropped on every retry.
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r26ca_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r26ca_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
: "${OBJ:?}"; : "${NFR:?}"; : "${MODE:?}"; : "${MESH:?}"; : "${GTDIR:?}"
cd ${T2}; echo "=== $(date) host=$(hostname) OBJ=${OBJ} nfr=${NFR} MODE=${MODE} ==="; nvidia-smi -L

${PY2} -u "${E}/rung26_mcfm_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets qkvo --blocks all --rank 4 --mcfm "${MODE}" \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"
echo "=== Done $(date) ==="
