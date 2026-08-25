#!/bin/bash
# Generalised build_targets: ONE object per job, orient file and suffix passed in.
# build_targets_one.sbatch hardcodes --orient out/orient/orientation_guan.json
# --suffix _guan, which is wrong for any object solved outside that file (the vase
# uses a KNOWN rotation, not a searched one, so it has its own orient file).
# Pass OBJ, ORIENT, SUFFIX. No training, no model load -- nvdiffrast only, minutes.
#SBATCH --job-name=mkt_obj
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mkt_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/mkt_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} OBJ=${OBJ} ORIENT=${ORIENT} SUFFIX=${SUFFIX} ==="
nvidia-smi -L

${PY2} -u "${E}/build_targets_new_objects.py" --res 960 --objects "${OBJ}" \
  --orient "${ORIENT}" ${SUFFIX:+--suffix "${SUFFIX}"} \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
