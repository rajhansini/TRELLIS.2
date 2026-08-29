#!/bin/bash
# fit_align: rescue an object whose mesh fails GATE-align.
#
# ONE object per job. Two steps, in order:
#   1. fit_mesh_to_video.py sweeps scale x dy and bakes the winner into the
#      canonical _render_frame.obj. This is the same repair that took alien_glow
#      from 43.03% uncovered to 0.05% on 2026-08-17.
#   2. build_targets_hero.py re-runs GATE-align on the result. If it passes it
#      writes gt_targets.json, which is the ONLY thing submit_on_targets.sh gates
#      on -- so a pass here releases the object's 14-cell ladder automatically.
#
# WHY IT READS FROM .prefit. The fit writes to the canonical path, so fitting a
# fitted mesh would compound the correction on every re-run. The pristine mesh is
# preserved once as <obj>_render_frame.obj.prefit and is always the input, which
# makes this job idempotent -- run it ten times and the result is identical.
#SBATCH --job-name=fit_align
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
# nvdiffrast has no sm_90 kernel in this build -> CUDA 209 on h002-class nodes.
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fit_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fit_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd ${T2}
MESH=${T2}/data/${OBJ}/mesh/${OBJ}_render_frame.obj
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} OBJ=${OBJ} ==="; nvidia-smi -L
[ -f "${MESH}.prefit" ] || { echo "MISSING ${MESH}.prefit — refusing to fit"; exit 1; }

echo "--- STEP 1: fit scale x dy against the video ---"
# --dy-range=-0.20,... uses '=' on purpose: a bare '-0.20' is parsed as a flag and
# argparse dies with 'expected one argument', which is how the first attempt failed.
${PY2} -u "${E}/fit_mesh_to_video.py" \
  --mesh "${MESH}.prefit" --raw \
  --frames-dir "${T2}/data/${OBJ}/frames_from_video" \
  --res 960 --probe-frames 1,75,150 \
  --s-range=0.60,1.00,21 --dy-range=-0.20,0.20,21 \
  --out "${MESH}" 2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "--- STEP 2: re-run GATE-align via build_targets_hero ---"
${PY2} -u "${E}/build_targets_hero.py" --obj "${OBJ}" --res 960 \
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== Done $(date) ==="
