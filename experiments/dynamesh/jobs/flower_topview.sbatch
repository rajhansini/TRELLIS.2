#!/bin/bash
# Camera sweep for the goat_flower gallery panel.
#
# The review asks for a top view where the flowers read, "rotated such that the artifact is
# not visible". Our rendered views for this object top out at diagC (yaw 225, elev +30), so
# there is no existing frame to crop: the view has to be rendered. Which yaw hides the
# defect is a looking question, not an arguing one, so this pins the figure frame and
# orbits the camera at a raised elevation, giving one sheet to choose from. Same convention
# as jobs/build_camera_options.py, which exists so cameras are chosen by eye.
#SBATCH --job-name=flowtop
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
set -euo pipefail
ELEV=${ELEV:-50}
PIN=${PIN:-82}
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
# NOT the MCFM run. --sweep angle loads ONLY the pinned frame into _raws (line 226),
# but the MCFM branch then builds conds for CFG['n_frames']=150 and indexes _raws[i-1],
# so any mcfm run + --sweep angle dies with IndexError in render_rung27_orbit.py:272.
# The camera angle does not depend on MCFM (it changes temporal blending, not pose), so
# the sweep runs on the plain rung27 arm. The chosen view is then rendered from the MCFM
# arm with --sweep both --turns 0, the fixed-camera path that diagA/diagC already use.
R=$E/runs/rung27_l1_lp_all_qkvo+sa_r4_s42_56b566f8
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd /net/projects/ranalab/guanc/geometry-def/dynamesh_render
echo "=== $(date) host=$(hostname) elev=${ELEV} pin=${PIN} ==="; nvidia-smi -L
$PY -u render_rung27_orbit.py \
  --run "$R" --ckpt lora_best.pt \
  --mesh /net/projects/ranalab/rajhansini/TRELLIS.2/data/goat_flower/mesh/goat_flower_render_frame.obj \
  --sweep angle --pin-frame ${PIN} --n-angles 12 --turns 1 \
  --elev ${ELEV} --radius 2.0 --res 518 \
  --tag goat_flower_top_e${ELEV}
echo "=== done $(date) ==="
