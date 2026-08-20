#!/bin/bash
# build_unseen_page.sh — encode any missing web videos, then build the four-camera page.
#
# FOUR CAMERAS, NOT SIX. The claim this page makes is "fitted to one camera, holds up
# at three it never saw", so it carries train + diagA + diagB + diagC. The two
# turntables are a coverage/seams artefact and belong on the six-camera page; putting
# them here doubles the byte budget for videos that do not support the claim.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
FF=/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin/ffmpeg
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
VIEWS=train,diagA,diagB,diagC
mkdir -p "$E/out/VIEWS_WEB"

# Encode at 1120 wide to match every video already in VIEWS_WEB -- mixing widths on
# one page makes the panels visibly different sizes row to row.
n=0
for v in train diagA diagB diagC; do
  for src in "$E"/out/VIEWS/*_${v}_GT_frozen_r27_mcfm.mp4; do
    [ -f "$src" ] || continue
    dst="$E/out/VIEWS_WEB/$(basename "$src")"
    [ -f "$dst" ] && continue
    "$FF" -nostdin -v error -y -i "$src" -vf scale=1120:-2 -c:v libx264 -crf 31 \
          -preset slow -pix_fmt yuv420p -movflags +faststart "$dst" \
      && { echo "encoded $(basename "$dst")"; n=$((n+1)); }
  done
done
echo "newly encoded: $n"

cd "$E" && "$PY2" jobs/build_views_artifact.py \
  "$E/out/unseen_angles.html" out/VIEWS_WEB "" "Unseen Angles" "$VIEWS"
