#!/bin/bash
# nightwatch.sbatch — survives this login session and the loss of any one node.
#
# Does the two things that would otherwise stop while Raj sleeps:
#   1. submit each new object's 14 rung cells the moment its targets land
#   2. run the watchdog, which resubmits infra failures and REFUSES GATE failures
#
# CPU only -- it submits and inspects, it never trains. Chains itself before the
# wall clock so a single --time cap cannot end the night.
#SBATCH --job-name=nightwatch
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/NIGHTWATCH.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/NIGHTWATCH.log
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd "$E"
echo "=== nightwatch up $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="

# Chain the successor FIRST. If this is left to the end, a node failure or an
# unexpected exit takes the whole chain down with it -- which is exactly how the
# previous orchestrator died at 08-18 07:15 with no successor.
sbatch --parsable --begin=now+225minutes jobs/nightwatch.sbatch > out/.nightwatch_next 2>&1 \
  && echo "successor queued: $(cat out/.nightwatch_next)"

END=$(( $(date +%s) + 225*60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  bash submit_on_targets.sh 2>&1 | sed "s/^/$(date +%H:%M)  /"
  $PY -u watchdog.py --minutes 4 --every 120 --since 2026-08-20T00:00 2>&1 \
    | grep -vE "^watchdog (up|exiting)" | sed "s/^/$(date +%H:%M)  wd  /"
  sleep 120
done
echo "=== nightwatch exiting $(date) — successor should take over ==="
