#!/bin/bash
# Front-band camera sweep for unicorn_effect1.
#
# The white blotch on the near shoulder is an UNPAINTED region: 6-7% of the silhouette in
# every front-facing render we have (train 7.4%, diagA 5.9%), against ~1% from behind
# (diagB 1.2%, diagC 1.0%). So it is not a bad frame, it is that patch of surface, and no
# existing camera both faces front and misses it. This orbits at a low elevation on one
# pinned frame so the yaw that keeps the front read while turning the patch away can be
# chosen by measurement rather than by guessing.
#SBATCH --job-name=unifront
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
ELEV=${ELEV:-12}
PIN=${PIN:-130}
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
R=$E/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_1a67b741
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd /net/projects/ranalab/guanc/geometry-def/dynamesh_render
echo "=== $(date) host=$(hostname) elev=${ELEV} pin=${PIN} ==="; nvidia-smi -L
$PY -u render_rung27_orbit.py \
  --run "$R" --ckpt lora_best.pt \
  --mesh /net/projects/ranalab/rajhansini/TRELLIS.2/data/unicorn_effect1/mesh/unicorn_effect1_render_frame.obj \
  --sweep angle --pin-frame ${PIN} --n-angles 24 --turns 1 \
  --elev ${ELEV} --radius 2.0 --res 518 \
  --tag unicorn_effect1_front_e${ELEV}_f${PIN}
echo "=== done $(date) ==="
