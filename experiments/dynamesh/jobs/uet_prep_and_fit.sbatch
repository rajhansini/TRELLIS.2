#!/bin/bash
# BATCH H prep + fit, chained.
#
# GATE-align rejected a hand-rolled mask by 22 px out of 187,915: my CPU rasteriser and
# nvdiffrast disagree on the silhouette edge, and the gate demands EXACT equality because a
# mismatch turns the texture loss into a shape loss. So the mask must be produced by
# make_render_mask.py, which uses the same rasteriser the trainer does.
#
# --raw is required: <obj>_render_frame.obj is already in the training frame, and
# preprocess() would re-normalise it (measured: control IoU 0.998 -> 0.311).
#
# Chained rather than three submissions because each step is minutes and the round trip is
# the expensive part.
#SBATCH --job-name=uet_prep
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/FIGRUNS/%x_%j.log
set -euo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=$T2/experiments/dynamesh
G=/net/projects/ranalab/guanc/geometry-def/dynamesh_render
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH
cd "$G"; echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L

MESH=$T2/data/unicorn_extended_texture/mesh/unicorn_extended_texture_render_frame.obj
echo "--- 1. mask, same rasteriser as the trainer ---"
$PY -u make_render_mask.py --mesh "$MESH" --res 960 --raw --tag uet_mask960
M=$G/out/uet_mask960/render_mask.npy
[ -s "$M" ] || { echo "FAILED: no mask at $M"; exit 1; }

echo "--- 2. 2D copies for both takes ---"
for o in unicorn_extended_texture unicorn_extended_texture_2; do
  $PY -u make_gt_targets.py --gt-dir "$T2/data/$o/frames_from_video" \
      --render-mask "$M" --render-res 960 --n-frames 150 --tag gt_targets_${o}_v2
  rm -rf "$E/out/gt_targets_$o"
  cp -r "$G/out/gt_targets_${o}_v2" "$E/out/gt_targets_$o"
  n=$(ls "$E/out/gt_targets_$o/frames"/*.png | wc -l)
  [ "$n" = 150 ] || { echo "FAILED: $o has $n frames"; exit 1; }
  echo "  $o: $n frames"
done

echo "--- 3. submit the two fits ---"
IDS=""
for o in unicorn_extended_texture unicorn_extended_texture_2; do
  J=$(sbatch --parsable --job-name=bd_${o} \
      --export=ALL,OBJ=$o,NFR=150,TARGETS=qkvo+sa,MODE=v2_D,\
MESH=$T2/data/$o/mesh/${o}_render_frame.obj,\
GTDIR=$E/out/gt_targets_${o}/frames \
      "$E/jobs/batch_d_arm.sbatch")
  echo "  $o -> $J"; IDS="$IDS $J"
done
echo "$IDS" | tr -s ' ' | sed 's/^ //' > "$E/out/FIGRUNS/batchH_jobids.txt"
echo "=== done $(date) ==="
