#!/bin/bash
# Fixed-azimuth render of unicorn_effect1, to find a front view without the unpainted
# shoulder patch (6-7% of the silhouette in train/diagA, ~1% from behind).
#
# NOT --sweep angle. That mode builds a single conditioning frame (_fr_list = [pin_frame]),
# but an MCFM checkpoint blends a WINDOW of conds and indexes _raws[0..n_frames-1], so it
# walks off a 1-element list: IndexError in cond_image. --sweep both with --turns 0 pins
# the azimuth across the whole sequence and loads every frame, which is what MCFM needs.
#SBATCH --job-name=unifix
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
set -euo pipefail
YAW=${YAW:?set YAW}; ELEV=${ELEV:-12}
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
R=$E/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1a67b741
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd /net/projects/ranalab/guanc/geometry-def/dynamesh_render
echo "=== $(date) host=$(hostname) yaw=${YAW} elev=${ELEV} ==="; nvidia-smi -L
$PY -u render_rung27_orbit.py \
  --run "$R" --ckpt lora_best.pt \
  --mesh /net/projects/ranalab/rajhansini/TRELLIS.2/data/unicorn_effect1/mesh/unicorn_effect1_render_frame.obj \
  --sweep both --turns 0 --yaw0 ${YAW} --elev ${ELEV} \
  --n-frames 150 --radius 2.0 --res 518 \
  --tag unicorn_effect1_y${YAW}_e${ELEV}
echo "=== done $(date) ==="
