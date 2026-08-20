#!/bin/bash
# Self-chaining watchdog. Runs on CPU only, submits its OWN successor before it
# exits, so it survives the 4h partition wall and any Claude session ending.
#
# It resubmits jobs that died for infrastructure reasons and refuses to resubmit
# jobs that died on a GATE — the latter need a fix, and retrying them just loops.
#
# To stop it:  scancel the running wd job AND `touch out/watchdog.stop`
#              (the stop file is what prevents the successor being submitted)
#SBATCH --job-name=wd
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/wd_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/wd_%j.log
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd "$E"
echo "=== watchdog $(date) job=${SLURM_JOB_ID} host=$(hostname) ==="

# Chain FIRST, not last: if this job is killed mid-run the successor already
# exists. A successor submitted at the end would never be submitted at all.
if [ ! -f "$E/out/watchdog.stop" ]; then
  NEXT=$(sbatch --parsable --begin=now+225minutes "$E/jobs/watchdog.sbatch" 2>&1)
  echo "successor queued: ${NEXT}"
else
  echo "out/watchdog.stop present — not chaining"
fi

${PY2} -u "$E/watchdog.py" --every 300 --minutes 225 --since 2026-08-17T05:00

echo "=== watchdog done $(date) ==="
