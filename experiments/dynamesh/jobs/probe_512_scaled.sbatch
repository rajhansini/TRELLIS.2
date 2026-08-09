#!/bin/bash
# Does implicit_gemm + grad-requiring decoder weights produce a NON-ZERO gradient
# at res 512 once LOSS_SCALE is applied? The unscaled probe returned exactly 0.0,
# which is the fp16 flush-to-zero signature, not a dead kernel.
#SBATCH --job-name=p512s
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/p512s_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/p512s_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
mkdir -p "${E}/out/spconv_probe"
for LS in 1 4096 65536; do
  echo ""; echo "######## implicit_gemm gradw res512 loss_scale=${LS} ########"
  ${PY2} -u "${E}/probe_spconv_bwd.py" --algo implicit_gemm --grad-weights \
    --resolution 512 --loss-scale ${LS} \
    --out "${E}/out/spconv_probe/r512_ls${LS}.json" \
    2>&1 | grep -E "PROBE|RESULT" | tail -5 || true
done
echo "=== Done $(date) ==="
