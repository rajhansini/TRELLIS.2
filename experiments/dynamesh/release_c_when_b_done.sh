#!/bin/bash
# Release batch C the moment batch B has no jobs left.
#
# WHY DETACHED. This ran as an in-session monitor and died with the session, leaving
# 90 C jobs held with nothing to release them -- the queue would have drained to
# empty and sat idle. setsid means it outlives the session.
#
# Releasing early is wrong (C would take B's slots back); releasing late is worse
# (idle GPUs). So it polls B and releases exactly at zero.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
HELD=/tmp/claude-27281/-net-projects-ranalab-rajhansini-TRELLIS/73637c7c-b537-4537-ae8e-20c14159ee56/scratchpad/held_C.txt
cd "$E"
echo "=== release watcher up $(date) pid=$$ ==="
while true; do
  B=$(squeue -u rajhansini -h -O 'Name:60,State:12' \
      | awk '$1 ~ /^(2[7-9]|3[0-4])_/ && $1 ~ /(animal_blob|chair_|napolean_|octopus_)/' | wc -l)
  if [ "$B" -eq 0 ]; then
    echo "$(date +%H:%M) batch B empty — releasing held C jobs"
    ids=$(squeue -u rajhansini -h -t PD -O 'JobID:12,Reason:20' | awk '$2=="JobHeldUser"{print $1}')
    [ -n "$ids" ] && scontrol release $ids && echo "$(date +%H:%M) released: $(echo $ids | wc -w)"
    # belt and braces: also release anything recorded at hold time
    [ -f "$HELD" ] && scontrol release $(cat "$HELD") 2>/dev/null
    echo "$(date +%H:%M) done, exiting"
    break
  fi
  echo "$(date +%H:%M) B still has $B jobs"
  sleep 300
done
