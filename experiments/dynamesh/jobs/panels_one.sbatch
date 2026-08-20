#!/bin/bash
# Composite the six panel videos for ONE object, on a compute node.
# The login node is shared and sits at load ~38; 150 frames x 4 panels x 6 views
# is real CPU work and does not belong there. Pass OBJ.
#SBATCH --job-name=pan
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/pan_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/pan_%j.log
set -o pipefail   # NOT -e: one failed view must not kill the other five
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
# ffmpeg/ffprobe are on /usr/bin on the LOGIN node only — compute nodes have
# neither, which is how five panel jobs died at the encode step with
# FileNotFoundError. The conda env carries both on shared storage, so put
# that on PATH rather than assuming a system binary exists.
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
cd "$E"; echo "=== $(date) host=$(hostname) OBJ=${OBJ} ==="
N=150; [ "${OBJ}" = "ancient_lady" ] && N=135
build () {  # view label supervised-flag
  V="$E/out/VIEWS/${OBJ}_$1_GT_frozen_r27_mcfm.mp4"
  if [ -f "$V" ]; then
    GOT=$(ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of csv=p=0 "$V" 2>/dev/null || echo 0)
    if [ "${GOT:-0}" -eq "$N" ]; then echo "skip $1 — already has $N frames"; return 0; fi
  fi
  ${PY2} -u "$E/build_view_panel.py" --obj "${OBJ}" --view "$1" --n "$N" --label "$2" $3 --out out/VIEWS
}
build train    "training view"        --supervised
build orbit360 "360° turntable"       ""
build orbit720 "720° turntable"       ""
build diagA    "yaw 45° · elev +25°"  ""
build diagB    "yaw 135° · elev −20°" ""
build diagC    "yaw 225° · elev +30°" ""
echo "=== done $(date) ==="
ls "$E/out/VIEWS/${OBJ}"_*_GT_frozen_r27_mcfm.mp4 2>/dev/null | wc -l | xargs -I{} echo "{}/6 panel videos present for ${OBJ}"
