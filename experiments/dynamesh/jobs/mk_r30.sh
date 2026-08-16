# $1 obj  $2 mesh  $3 gtdir  $4 targets  $5 nframes  $6 tag  $7 extra-flags  $8 epochs
cat <<SB
#!/bin/bash
#SBATCH --job-name=$6_$1
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S|a30"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/$6_$1_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/$6_$1_%j.log
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=\${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_\${SLURM_JOB_ID}.json
cp -f "\$HOME/.flex_gemm/autotune_cache.json" "\$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd \${T2}
echo "=== \$(date) host=\$(hostname) job=\${SLURM_JOB_ID} ==="; nvidia-smi -L
echo "### obj=$1 mcfm=v2_D frames=$5 $7"
\${PY2} -u "\${E}/rung30_kl_both.py" \\
  --mesh "$2" --gt-dir "$3" --gt-render-dir "$4" \\
  --render-res 960 --resolution 512 \\
  --targets qkvo+sa --blocks all --rank 4 \\
  --recon l1 --lpips --w-lpips 0.1 \\
  --mcfm v2_D \\
  --kl --kl-blocks all --kl-voxels 256 $7 \\
  --epochs $8 --n-frames $5 --seed 42 --lr 1e-4 \\
  2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|^\s+@torch|^\s+_TORCH|^\s+def backward|it/s\]"
echo "=== Done \$(date) ==="
SB
