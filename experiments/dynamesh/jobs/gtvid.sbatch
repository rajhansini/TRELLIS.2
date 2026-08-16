#!/bin/bash
# GT video frame vs the 2D copy, every frame, one object per job.
# CPU only — no GPU, no model. Pass OBJ=<name>.
#SBATCH --job-name=gtvid
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gtvid_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/gtvid_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) OBJ=${OBJ} ==="
${PY2} -u "${E}/render_gt_vs_target_video.py" --obj "${OBJ}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
