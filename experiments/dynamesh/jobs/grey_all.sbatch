#!/bin/bash
# grey_all.sbatch — the grey 960^2 front render of every mesh, for the video model.
#
# WHY: only spot and teapot ever had one (out/{spot,teapot}_grey/*_front_960.png).
# The six Guan objects were never rendered on our side because Guan supplied their
# videos, so the only images on disk for them are out/guan_check/*_guan.png — 512^2
# flat-shaded PIL painter's-algorithm panels from cpu_align_check.py, built to judge
# ORIENTATION, not to be conditioning images. This produces the real thing for all
# eight, through the same render_grey.py the two good ones came from.
#
# NORMALISATION IS DECIDED BY MEASUREMENT, NOT ASSUMPTION. render_grey.py recentres
# and rescales the mesh unless --raw is passed, and the *_render_frame_guan.obj files
# are already baked into the render frame, so the correct flag is not obvious from
# the name. Each object is therefore rendered BOTH ways and the silhouette is counted
# against that object's own render_mask.npy — the exact mask the training run used.
# Whichever agrees within 1% is kept, the other is deleted. A mesh where NEITHER
# agrees is reported and left out rather than silently shipped, because a render at
# the wrong scale looks perfectly fine on its own and only fails later.
#
# nvdiffrast has kernel images for sm86 (A40) and sm89 (L40S) only; a bare
# --gres=gpu:1 can land on an A30 or rtx2080ti and die with "no kernel image".
#SBATCH --job-name=grey_all
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|L40S"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/grey_all_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/grey_all_%j.log
set -uo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export HF_HOME=/net/scratch/rajhansini/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPCONV_ALGO=native
OUT=${E}/out/GREY_RENDERS; mkdir -p "$OUT"
cd ${T2}
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID} ==="; nvidia-smi -L

#      object                mesh                                            mask dir
ROWS="
horse_metal|${T2}/data/horse_metal/mesh/horse_metal_render_frame_guan.obj|gt_targets_horse_metal_guan
penguin_circuits|${T2}/data/penguin_circuits/mesh/penguin_circuits_render_frame_guan.obj|gt_targets_penguin_circuits_guan
pumpkin_rot|${T2}/data/pumpkin_rot/mesh/pumpkin_rot_render_frame_guan.obj|gt_targets_pumpkin_rot_guan
teapot_ceramic_crack|${T2}/data/teapot_ceramic_crack/mesh/teapot_ceramic_crack_render_frame_guan.obj|gt_targets_teapot_ceramic_crack_guan
teapot_porcelain|${T2}/data/teapot_porcelain/mesh/teapot_porcelain_render_frame_guan.obj|gt_targets_teapot_porcelain_guan
whale_spots|${T2}/data/whale_spots/mesh/whale_spots_render_frame_guan.obj|gt_targets_whale_spots_guan
spot_lava|${T2}/data/spot_lava/mesh/spot_render_frame.obj|gt_targets_spot_lava
teapot_lava2|/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply|gt_targets_teapot_lava2
"

PASS=0; FAIL=0
for row in $ROWS; do
  OBJ=${row%%|*}; rest=${row#*|}; MESH=${rest%%|*}; MDIR=${rest##*|}
  REF=${T2}/data/${OBJ}/frames_from_video/frame_0001.png
  MASK=${E}/out/${MDIR}/render_mask.npy
  echo ""; echo "================ ${OBJ} ================"
  [ -f "$MESH" ] || { echo "  SKIP: no mesh at $MESH"; FAIL=$((FAIL+1)); continue; }
  [ -f "$MASK" ] || { echo "  SKIP: no render_mask at $MASK"; FAIL=$((FAIL+1)); continue; }
  [ -f "$REF" ]  || { echo "  note: no ref frame, rendering untoned"; REF=""; }

  TGT=$($PY2 -c "import numpy as np;print(int(np.load('$MASK').sum()))")
  echo "  target silhouette ${TGT} px  (from ${MDIR}/render_mask.npy)"

  BEST=""; BESTPX=0
  for MODE in norm raw; do
    TMP=${OUT}/.${OBJ}_${MODE}.png
    ARGS=""; [ "$MODE" = "raw" ] && ARGS="--raw"
    [ -n "$REF" ] && ARGS="$ARGS --ref $REF"
    $PY2 -u "${E}/render_grey.py" --mesh "$MESH" --res 960 --out "$TMP" $ARGS 2>&1 | sed 's/^/    /'
    [ -f "$TMP" ] || { echo "    ${MODE}: render produced nothing"; continue; }
    PX=$($PY2 -c "
from PIL import Image; import numpy as np
a=np.asarray(Image.open('$TMP').convert('L'))
print(int((a<250).sum()))")
    DEV=$($PY2 -c "print(f'{abs($PX-$TGT)/max($TGT,1)*100:.2f}')")
    echo "    ${MODE}: ${PX} px   deviation ${DEV}%"
    OK=$($PY2 -c "print(1 if abs($PX-$TGT)/max($TGT,1) <= 0.01 else 0)")
    if [ "$OK" = "1" ]; then BEST=$MODE; BESTPX=$PX; fi
  done

  if [ -n "$BEST" ]; then
    mv -f "${OUT}/.${OBJ}_${BEST}.png" "${OUT}/${OBJ}_grey_front_960.png"
    rm -f ${OUT}/.${OBJ}_*.png
    echo "  PASS  kept ${BEST}  ->  ${OBJ}_grey_front_960.png  (${BESTPX} px vs ${TGT})"
    PASS=$((PASS+1))
  else
    echo "  FAIL  neither norm nor raw matched the training silhouette — NOT shipped."
    echo "        A wrong-scale render looks fine alone and only fails downstream."
    mkdir -p ${OUT}/_rejected && mv -f ${OUT}/.${OBJ}_*.png ${OUT}/_rejected/ 2>/dev/null
    FAIL=$((FAIL+1))
  fi
done

echo ""; echo "======================================================"
echo "  PASS ${PASS}   FAIL ${FAIL}"
ls -la ${OUT}/*.png 2>/dev/null
echo "=== Done $(date) ==="
