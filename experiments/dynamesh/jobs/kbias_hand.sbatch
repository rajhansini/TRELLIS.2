#!/bin/bash
# kbias_hand.sbatch — does biasing the training-timestep draw cut training time?
#
# Runs rung27_fastprefix.py, a COPY. rung27_selfattn_lora.py is untouched and
# every existing run directory is untouched: k_bias is in the config dict, so the
# md5 run-id differs and these land in new directories.
#
# Each training step pays k no-grad evals to reach a random knot k of the 12-step
# schedule, plus 1 graded eval. Uniform k gives E[k]=5.5, i.e. 6.5 evals/step.
#   --k-bias p  ->  k = int(STEPS * u**p),  E[k] = STEPS/(p+1)
#     p=1  6.5 evals/step  (uniform, the control)
#     p=2  5.0 evals/step  ~1.30x
#     p=3  4.0 evals/step  ~1.63x
#
# THE TRADE, stated so no one reads the speedup alone: T_SEQ runs 1->0, so low k
# is the HIGH-NOISE end. Biasing k down trains the adapter less often at the
# low-noise timesteps that carry fine detail. Judge on final PSNR, not wall time.
# fig27: rung27_fastprefix.py
# Every path is a PARAMETER. Hardcoding hero paths would silently train the
# guan-pipeline objects (pumpkin_rot) against the wrong targets at the wrong
# frame count, complete successfully, and be wrong.
# One log per run, named by object+arm, not just jobid.
#SBATCH --job-name=kbias
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G      # measured peak RSS 15.3G. 96G cost 24 of 34 billing units
#                      # under the group 1024 TRES cap, capping concurrency at 30.
#SBATCH --time=04:00:00   # partition MaxTime is 04:00:00 on ALL partitions -- a longer
#                          # request is REJECTED at submit. 30 epochs takes 3:33-3:52,
#                          # so timeouts are unavoidable; resume-from-checkpoint covers it.
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
cd ${T2}
echo "=== $(date) host=$(hostname) OBJ=${OBJ} ARM=fig27 MODE=${MODE:-none} nfr=${NFR} ==="
echo "    MESH=${MESH}"
echo "    GTDIR=${GTDIR}"
nvidia-smi -L
${PY2} -u "${E}/rung27_fastprefix.py" \
  --mesh "${MESH}" --gt-dir "${T2}/data/${OBJ}/frames_from_video" \
  --gt-render-dir "${GTDIR}" \
  --render-res 960 --resolution 512 \
  --targets qkvo+sa --blocks all --rank 4 \
  --recon l1 --lpips --w-lpips 0.1 \
  --epochs 30 --n-frames "${NFR}" --seed 42 --lr 1e-4 \
   ${MODE:+--mcfm ${MODE}} ${KBIAS:+--k-bias ${KBIAS}} 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
