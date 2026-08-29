#!/bin/bash
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
cd "$E"
echo "=== release watcher up $(date) pid=$$ ==="
while true; do
  R4=$(squeue -u rajhansini -h -O 'Name:44' | grep -c '^rarm4_')
  if [ "$R4" -eq 0 ]; then
    echo "$(date +%H:%M) rarm4 empty — releasing held rarm7c jobs"
    ids=$(squeue -u rajhansini -h -t PD -O 'JobID:12,Reason:20' | awk '$2=="JobHeldUser"{print $1}')
    [ -n "$ids" ] && scontrol release $ids && echo "$(date +%H:%M) released: $(echo $ids | wc -w)"
    echo "$(date +%H:%M) done, exiting"
    break
  fi
  echo "$(date +%H:%M) rarm4 still has $R4 jobs"
  sleep 300
done
