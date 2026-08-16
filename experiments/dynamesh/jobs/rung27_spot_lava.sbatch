#!/bin/bash
# rung27 on spot_lava — LoRA on SELF-ATTENTION in addition to cross-attention.
#
# rung27, not 26: rung20_conf_lora.py already maps the key-locked 'v' arm to 26
# and kvo to 25, so self-attention takes the next free number.
#
# WHY THIS RUN EXISTS: rungs 17/18/19/25 all move the adapter WITHIN
# cross-attention (kv / qkv / kvo / qkvo). That is an intra-module ablation. The
# paper claims cross-attention is the right PLACE, and defending that needs a
# cross-MODULE arm. self-attention is the only one worth running, because it is
# the propagation mechanism every other rung deliberately froze
# (rung20_conf_lora.py:516 — "leaving it alone is the whole design").
#
# MATCHED CELL. Byte-identical to jobs/rung25_spot_lava.sbatch except --targets:
#   --recon l1 --lpips --w-lpips 0.1   (same loss as 17/18/19/25 in the grid)
#   --render-res 960 --resolution 512  (same pixels scored)
#   --rank 4 --epochs 30 --seed 42 --lr 1e-4 --blocks all
#   NO --conf-weight: the trust map lost 5/5 matched pairs and every arm after
#   rung20 is unweighted. Passing it would also force _RUNG to 20 and mislabel
#   this run.
# Anything else would not be comparable and could not enter the ladder table.
#
# CONFIG VERIFIED before submission by executing the script's own logic:
#   rung=26  targets=qkvo+sa  target_set=(to_q,to_kv,to_out,sa_qkv,sa_out)
#   150 B matrices  2,334,720 params   (rung19 qkvo = 90 mats, 1,228,800)
#
# PREDICTION ON RECORD (see the script docstring): front-view PSNR should come
# out ABOVE rung19 simply because there are more parameters and the ladder is
# monotone in adapted-parameter count with no plateau. The verdict is the
# TURNTABLE, not this PSNR. If the orbit is also better, the design premise is
# wrong and that is the finding.
#
# Targets already exist and are NOT rebuilt — same ones rung17/19/25 used:
#   gt_targets_spot_lava/frames  150 gt_*.png  150 valid_*.npy  render_mask.npy 960^2
#
# SMOKE: sbatch --time=00:40:00 --export=ALL,SMOKE=1 jobs/rung27_spot_lava.sbatch
#   Same arguments, 2 epochs / 8 frames. Different config hash, so it lands in
#   its own directory and can never resume into the full run's checkpoints.
#SBATCH --job-name=r27_spot
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 4 h is the partition cap; checkpoints every epoch and --requeue resumes,
# guarded by GATE-resume on mesh / targets / render_res.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27_spot_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27_spot_%j.log
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
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} smoke=${SMOKE:-0} ==="; nvidia-smi -L

${PY2} -u "${E}/rung27_selfattn_lora.py" \
  --mesh "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/mesh/spot_render_frame.obj" \
  --gt-dir "/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video" \
  --gt-render-dir "${E}/out/gt_targets_spot_lava/frames" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames 150 --seed 42 --lr 1e-4 \
  ${SMOKE:+--smoke} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward"

echo "=== Done $(date) ==="
