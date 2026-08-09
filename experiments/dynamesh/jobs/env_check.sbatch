#!/bin/bash
#SBATCH --job-name=t2_envcheck
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/envcheck_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/envcheck_%j.log
set -eo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
echo "=== $(date) host=$(hostname) ==="
nvidia-smi -L
echo "compute node glibc: $(ldd --version | head -1)"
cd "$D"
${PY2} - <<'PYEOF'
import sys; sys.path.insert(0, '.')
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
import o_voxel; print('o_voxel OK')
from trellis2.pipelines import Trellis2TexturingPipeline
print('Trellis2TexturingPipeline OK')
import inspect
print('run() signature:', inspect.signature(Trellis2TexturingPipeline.run))
PYEOF
echo "=== Done $(date) ==="
