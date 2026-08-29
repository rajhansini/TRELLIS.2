#!/bin/bash
# Detached login-node twin of jobs/nightwatch.sbatch. Starts immediately instead of
# waiting behind Priority in the general queue. Safe to run alongside the slurm one:
# submit_on_targets.sh takes an atomic mkdir lock per object.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd "$E"
echo "=== nightwatch_local up $(date) pid=$$ ==="
while true; do
  bash submit_on_targets.sh 2>&1 | sed "s/^/$(date +%H:%M)  /"
  timeout 300 $PY -u watchdog.py --minutes 4 --every 120 --since 2026-08-20T00:00 2>&1 \
    | grep -vE "^watchdog (up|exiting)" | sed "s/^/$(date +%H:%M)  wd  /"
  sleep 120
done
