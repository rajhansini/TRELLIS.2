#!/bin/bash
# Texel metrics for a run whose directory name is not known at submit time.
#
# WHY THIS IS NOT texel_any.sbatch. That one takes RUN=<dir>. A run launched in the
# same batch has a config-hash directory name that does not exist until the trainer
# writes it, so a dependent job cannot be given RUN up front. This resolves the run
# from (RUNG, MODE, OBJ) by reading each candidate's own config.json -- never by
# picking the newest directory, which would grab another object's run if two land
# together.
#SBATCH --job-name=texdep
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/texdep_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/texdep_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
export FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm_${SLURM_JOB_ID}.json
cp -f "$HOME/.flex_gemm/autotune_cache.json" "$FLEX_GEMM_AUTOTUNE_CACHE_PATH" 2>/dev/null || true
cd /net/projects/ranalab/rajhansini/TRELLIS.2
: "${OBJ:?}"; : "${MODE:?}"; : "${TAG:?}"; : "${NFR:?}"; : "${RUNG:?}"
echo "=== $(date) host=$(hostname) OBJ=${OBJ} MODE=${MODE} TAG=${TAG} nfr=${NFR} RUNG=${RUNG} ==="
OUTJ="${E}/out/TEXEL/${TAG}.json"
if [ -s "$OUTJ" ]; then echo "ALREADY DONE: $OUTJ"; exit 0; fi
RUN=$(${PY2} - "$E" "$OBJ" "$MODE" "$RUNG" <<'PYX'
import json,sys,glob,os
E,obj,mode,rung=sys.argv[1:5]
hit=[]
for c in glob.glob(f'{E}/runs/rung{rung}_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if str(d.get('mcfm'))!=mode: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if not os.path.exists(os.path.join(os.path.dirname(c),'ckpts','lora_best.pt')): continue
    hit.append(os.path.dirname(c))
if len(hit)!=1:
    sys.exit(f'expected exactly 1 rung{rung} {mode} run for {obj} with a checkpoint, found {len(hit)}: {hit}')
print(hit[0])
PYX
)
echo "resolved RUN=${RUN}"
[ -f "${RUN}/config.json" ] || { echo "no config.json under ${RUN}"; exit 1; }
mkdir -p "${E}/out/TEXEL"
nvidia-smi -L
# Renderer is chosen by RUNG. A rung37 checkpoint carries blocks_t.*, gates.* and
# tmix.*; rung27's registry has none of them and load_state_dict dies with
# "Unexpected key(s)" before the first frame. rung31 has blocks_t and gates but no
# tmix, so it does not substitute either. Hardcoding rung27 here is what silently
# broke the whole rung37 texel batch.
case "${RUNG}" in
  37) ORBIT=render_rung37_orbit.py ;;
  31) ORBIT=render_rung31_orbit.py ;;
  *)  ORBIT=render_rung27_orbit.py ;;
esac
echo "[ORBIT] rung=${RUNG} -> ${ORBIT}"
${PY2} -u "${E}/${ORBIT}" --run "${RUN}" --sweep both \
  --n-frames "${NFR}" --n-angles "${NFR}" --turns 0 --skip-render \
  --texel-metrics "${OUTJ}" --tag "texel_${TAG}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn|it/s\]"
[ -s "$OUTJ" ] || { echo "FAILED: no json written"; exit 1; }
echo "=== Done $(date) ==="
