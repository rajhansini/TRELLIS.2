#!/bin/bash
set -eo pipefail
cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
for i in $(seq 1 240); do
  D=0; N=0
  for E in 30 45; do for Y in 0 45 90 135 180 225 270 315; do
    N=$((N+1)); C=0; F=out/view_ivysaur_petal_27g_top${E}y${Y}/frames
    [ -d "$F" ] && C=$(find "$F" -maxdepth 1 -name '*.png' | wc -l)
    [ "$C" -ge 150 ] && D=$((D+1))
  done; done
  [ "$D" -eq "$N" ] && { echo "all $N renders complete"; break; }
  Q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^tv_ivy' || true)
  [ "$Q" -eq 0 ] && [ "$i" -gt 3 ] && { echo "QUEUE EMPTY with $D/$N complete"; break; }
  sleep 30
done
