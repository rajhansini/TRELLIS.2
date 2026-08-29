#!/bin/bash
#SBATCH --job-name=frde
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frde_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/frde_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) OBJ=${OBJ} ==="
$PY -u "$E/jobs/fullrate_metrics.py" --obj "$OBJ" --out "$E/out/FULLRATE/${OBJ}.json"
echo "=== Done $(date) ==="
