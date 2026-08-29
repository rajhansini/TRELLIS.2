#!/bin/bash
# Batch-C panel build: the FOUR canonical views only.
# panels_one.sbatch tries six (adds orbit360/orbit720). Batch C was rendered at the
# fixed training view and three diagonals only, so the two turntables would fail on a
# missing directory every run and bury the real errors. Pass OBJ.
#SBATCH --job-name=panc
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/panc_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/panc_%x_%j.log
set -o pipefail   # NOT -e: one failed view must not kill the other three
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
cd "$E"; echo "=== $(date) host=$(hostname) OBJ=${OBJ} ==="
N=150
build () {
  V="$E/out/VIEWS/${OBJ}_$1_GT_frozen_r27_mcfm.mp4"
  if [ -f "$V" ]; then
    GOT=$(ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of csv=p=0 "$V" 2>/dev/null || echo 0)
    if [ "${GOT:-0}" -eq "$N" ]; then echo "skip $1 — already has $N frames"; return 0; fi
  fi
  ${PY2} -u "$E/build_view_panel.py" --obj "${OBJ}" --view "$1" --n "$N" --label "$2" $3 --out out/VIEWS
}
build train "training view"       --supervised
build diagA "yaw 45° · elev +25°"  ""
build diagB "yaw 135° · elev −20°" ""
build diagC "yaw 225° · elev +30°" ""
echo "=== done $(date) ==="
ls "$E/out/VIEWS/${OBJ}"_*_GT_frozen_r27_mcfm.mp4 2>/dev/null | wc -l | xargs -I{} echo "{}/4 panel videos present for ${OBJ}"
