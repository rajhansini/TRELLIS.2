#!/bin/bash
# submit_video_comparison.sh -- fixed-camera renders for the VIDEO-SPACE comparison table.
#
# WHY. Table 1's Flicker/Accel/Drift columns are texel-space, computed on the PBR voxel
# field. The competing methods do not produce a 3D field, so those columns can never carry
# a baseline row. Recomputing them from rendered video makes all six rows comparable.
# Table 2 (the ablation) is unaffected -- every row there is ours and is 3D.
#
# ONE JOB PER (object, arm). render_rung27_orbit.py composites [frozen | adapted] into one
# canvas per frame, so the frozen row and the adapted row share a camera by construction and
# video_metrics.py --panel crops them apart with a layout assert.
#
# RUNS ARE PINNED from out/video_arm_map.json, chosen highest-PSNR where an object has
# several runs of one config. That rule only fires on spot_lava, whose alternates are failed
# runs (19.73 and 11.81 dB against 23.99) rather than samples of the same thing.
#
# RES=518 matches the spot_lava pilot. THE BASELINES MUST BE RENDERED THROUGH THIS SAME
# CAMERA or none of it is comparable -- run_baseline.py's existing output is a different
# camera AND a different object (f0075), silhouette IoU 0.386 against ours.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
MAP=$E/out/video_arm_map.json
RES=518
n=0; skipped=0
while IFS=$'\t' read -r OBJ ARM RUN NFR PSNR; do
  TAG="${OBJ}_${ARM}"
  if [ -d "$E/out/fixview_${TAG}/frames" ] && \
     [ "$(ls "$E/out/fixview_${TAG}/frames"/*.png 2>/dev/null | wc -l)" -eq "$NFR" ]; then
    echo "  skip $TAG (already rendered)"; skipped=$((skipped+1)); continue
  fi
  EXP="RUN=runs/${RUN},TAG=${TAG},NFR=${NFR},RES=${RES}"
  J=$(sbatch --parsable --job-name="fxv_${TAG}" --export=ALL,"$EXP" "$E/jobs/fixview_arm.sbatch")
  python3 - "$MAN" "$J" "fxv_${TAG}" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/fixview_arm.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$J" "fxv_${TAG}" \
    "VIDEO-SPACE comparison: fixed-camera frames (--turns 0), ${OBJ} arm=${ARM} rung27 run ${RUN##*_} (PSNR ${PSNR}), ${NFR}fr at ${RES}px; canvas is [frozen | adapted] so both rows share a camera; feeds video_metrics.py --panel 0/1; out=$E/out/fxv_${J}.log; PASS: $E/out/fixview_${TAG}/frames holds ${NFR} pngs"
  echo "  $J  fxv_${TAG}  ${ARM}  ${RUN##*_}"
  n=$((n+1))
done < <(python3 - "$MAP" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
for k in sorted(m):
    o,a=k.split('|'); v=m[k]
    print(f"{o}\t{a}\t{v['run']}\t{v['nfr']}\t{v['psnr']:.3f}")
PY
)
echo "submitted $n, skipped $skipped"
