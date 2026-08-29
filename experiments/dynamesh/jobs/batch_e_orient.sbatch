#!/bin/bash
# STEP 2 for the six unaligned objects: search for the rotation that puts each
# mesh where its video is. 24 cube rotations coarse, then coordinate descent on
# yaw/pitch/roll. spot_lava is searched too as a control and must come back
# unchanged (GATE-solve).
# the identical code path. No training, no model load — nvdiffrast only, minutes.
#SBATCH --job-name=orient_be
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orient_be_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orient_be_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

${PY2} -u "${E}/solve_orientation.py" --res 960 --search-res 384 --frame 1 --jobs-file "${E}/jobs/batch_e_orient.json" --out "${E}/out/orient_batch_e" \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
