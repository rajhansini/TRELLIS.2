#!/bin/bash
# BATCH D, one arm of the 7-arm supplementary ladder, one object per job.
# TARGETS selects the rung: qkvo = rung19 (cross-attn only), qkvo+sa = rung27
# (adds self-attention). Both run rung27_selfattn_lora.py -- rung19_guan.sbatch
# does the same, and only rung27_selfattn_lora.py implements --mcfm at all.
# MODE is the MCFM variant, or EMPTY for no blend: ${MODE:+--mcfm} omits the flag
# entirely rather than passing an empty string, which argparse would take as a mode.
# v2_D chosen because it won on BOTH already-trained objects and the ordering was
# identical on each: spot_lava v2_D 24.194 > v2_C 24.061 >> v3_C 22.668 ~ v3_D 22.659;
# teapot_lava2 v2_D 22.688 > v2_C 22.675 >> v3_D 20.336 ~ v3_C 20.131.
# MCFM blends the cached DINOv3 conditioning temporally before training; it needs
# no per-object precompute and derives its window from conds.keys(), so 121 frames
# behaves exactly like 150.
# One object per job. Pass OBJ and NFR.
#   --mesh           data/<obj>/mesh/<obj>_render_frame.obj   pose baked in
#   --gt-dir         data/<obj>/frames_from_video                  conditioning
#   --gt-render-dir  out/gt_targets_<obj>/frames              the 2D copy
# GATE-align asserts our training render and the target mask are the SAME pixels,
# so a mesh/target mismatch cannot slip through silently.
#SBATCH --job-name=bd_arm
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# a30 REMOVED. Every crashed st_D/v3_D run on disk died with `GLIBC_2.32 not found`
# importing o_voxel._C -- the trellis2 env is built for a40/L40S only. This must live
# in the SCRIPT, not on the submit command line: watchdog.py resubmits with only
# --job-name and --export, so a CLI --constraint is silently dropped on every retry,
# and 30 epochs against a 4h wall means retries are the normal case, not the rare one.
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
# 96G -> 32G. Same script and same flags as fig27.sbatch, whose measured peak RSS is
# 15.3G; 96G costs 24 of 34 billing units under the group TRES cap and throttled the
# fleet to ~30 concurrent for nothing. Also dropped by a watchdog resubmit if it were
# only a CLI flag, so it belongs here too.
#SBATCH --mem=32G
# 4h is the HARD CAP on general, threedle-contrib and threedle-own alike -- 6h is
# rejected at submit time, so this cannot be raised. That matters here: finished
# batch A-C arms measured 3:03-3:59 elapsed at this wall (napolean_teapot_crack
# landed at 3:58 and 3:59), and batch D trains 150 frames where many A-C objects
# had 121. So most of these 77 WILL time out and must resume. That path is wired:
# --requeue below, per-epoch ckpts/lora_e*.pt written atomically, the trainer scans
# them newest-first with a torn-file fallback, GATE-resume refuses a checkpoint from
# a different mesh, and 'bd_' is registered in watchdog.py's KIND table so a TIMEOUT
# is resubmitted with its manifest --export rather than sitting dead.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/bd_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/bd_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd ${T2}; echo "=== $(date) host=$(hostname) OBJ=${OBJ} nfr=${NFR} TARGETS=${TARGETS} MODE=${MODE:-none} MESH=${MESH} GTDIR=${GTDIR} ==="; nvidia-smi -L

${PY2} -u "${E}/rung27_selfattn_lora.py" \
  --mesh "${MESH}" \
  --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets "${TARGETS}" --blocks all --rank 4 ${MODE:+--mcfm "${MODE}"} \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"
echo "=== Done $(date) ==="
