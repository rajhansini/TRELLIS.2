#!/bin/bash
#SBATCH --job-name=axdale
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/axdale_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/axdale_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1
cd /net/projects/ranalab/rajhansini/TRELLIS.2
$PY2 -u "$E/axis_sheet_dale.py" dragons_xyzrgb_dragon statues_napoleon thingi10k_octocat \
  thingi10k_wooly_sheep humanoid_mike_wazowski animals_octopus furnature_chair \
  vehicles_tie_fighter animals_blub animals_bob aliens_ufo 2>&1 \
  | grep -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== Done $(date) ==="
