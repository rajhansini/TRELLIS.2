#!/bin/bash
# Self-chaining orchestrator. Chains its successor FIRST so a mid-run kill still
# leaves one queued. Stop with: touch out/orchestrate.stop  (and scancel).
#SBATCH --job-name=orch
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orch_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/orch_%j.log
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
# ffmpeg/ffprobe are on /usr/bin on the LOGIN node only — compute nodes have
# neither, which is how five panel jobs died at the encode step with
# FileNotFoundError. The conda env carries both on shared storage, so put
# that on PATH rather than assuming a system binary exists.
export PATH="/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin:$PATH"
cd "$E"; echo "=== orchestrator $(date) job=${SLURM_JOB_ID} host=$(hostname) ==="
if [ ! -f "$E/out/orchestrate.stop" ]; then
  sbatch --parsable --begin=now+225minutes "$E/jobs/orchestrate.sbatch" | xargs -I{} echo "successor {}"
fi
${PY2} -u "$E/orchestrate.py" --every 300 --minutes 225
echo "=== done $(date) ==="
