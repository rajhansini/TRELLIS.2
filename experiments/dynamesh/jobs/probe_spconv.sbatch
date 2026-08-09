#!/bin/bash
# Which sparse-conv config can backprop through TRELLIS.2's texture decoder?
# Each candidate runs in its OWN process — the algorithm is a module global and
# the neighbour caches persist, so one process per config is the only honest test.
#SBATCH --job-name=probe_spc
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:a40:1   # PIN the GPU type: results are memory-sensitive and the
#                          # cluster mixes 10.5 GiB 2080 Ti with 44.5 GiB A40.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/probespc_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/probespc_%j.log
set -eo pipefail
ROOT=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${ROOT}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPCONV_ALGO=native
cd /net/projects/ranalab/rajhansini/TRELLIS.2
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="
nvidia-smi -L

mkdir -p "${E}/out/spconv_probe"

run () {  # $1 algo   $2 "" | --grad-weights   $3 tag
  echo ""
  echo "######## ${3} ########"
  ${PY2} -u "${E}/probe_spconv_bwd.py" --algo "$1" $2 --resolution "${RES:-1024}" \
     --out "${E}/out/spconv_probe/r${RES:-1024}_${3}.json" \
     2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn" | tail -25 || true
}

run explicit_gemm                ""              explicit_frozen
run implicit_gemm                --grad-weights  implicit_gradw
run implicit_gemm_splitk         --grad-weights  implicitsk_gradw
run masked_implicit_gemm_splitk  --grad-weights  maskedsk_gradw

echo ""
echo "######## SUMMARY ########"
${PY2} - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob('/net/projects/ranalab/rajhansini/TRELLIS/experiments/'
                          'trellis2_baseline/out/spconv_probe/*.json')):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[:-5], d))
print(f'{"config":22}{"ok":>5}{"peak GiB":>10}{"sec":>8}  error')
for name, d in rows:
    print(f'{name:22}{str(d["ok"]):>5}{str(d["peak_gib"]):>10}'
          f'{str(d["seconds"]):>8}  {(d["error"] or "")[:80]}')
EOF
echo "=== Done $(date) ==="
