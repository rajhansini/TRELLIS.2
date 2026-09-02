#!/bin/bash
set -eo pipefail
cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
for i in $(seq 1 200); do
  D=0
  for Y in 0 45 90 135 180 225 270 315; do
    C=0; F=out/view_ivysaur_petal_27g_top60y$Y/frames
    [ -d "$F" ] && C=$(find "$F" -maxdepth 1 -name '*.png' | wc -l)
    [ "$C" -ge 150 ] && D=$((D+1))
  done
  [ "$D" -eq 8 ] && { echo "all 8 top-view renders complete"; break; }
  Q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^tv_ivy' || true)
  [ "$Q" -eq 0 ] && [ "$i" -gt 3 ] && { echo "QUEUE EMPTY with only $D/8 complete"; break; }
  sleep 30
done
/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python jobs/topview_contact.py
