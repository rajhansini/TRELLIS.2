#!/bin/bash
# Texel metrics for the 54 batch D+E cells (18 objects x r19 / r27 / r27mcfm).
# PSNR and SSIM come from each run's final_eval.json; flicker, acceleration and drift
# do not exist until this runs, so no table can be built before it.
# texel_after_train.sbatch resolves the run dir from (RUNG, MODE, OBJ) by reading each
# candidate's own config.json -- never by newest-directory, which would grab another
# object's run when two land together.
# Idempotent: a cell whose out/TEXEL/<tag>.json already exists, or that is queued, is skipped.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json; JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="moai_silver moai_animated gargoyle_spiral gargoyle_effect_one airplane_blub airplane_red_cracks teddy_bleach teddy_fusion goat_burnt goat_flower goat_clay ivysaur_petal ivysaur_petal_2 blub_raurshaw blub_drying pegasus_shine pegaso_effect_1 mosaic_painting"
# tag : RUNG : MODE   (MODE is matched against str(config['mcfm']), so 'None' not '')
ARMS="r19:19:None r27:27:None r27mcfm:27:v2_D"
squeue -u "$USER" -h -o "%j" > /tmp/_de_q.$$ || true
N=0; SK=0
for OBJ in $OBJS; do
  for A in $ARMS; do
    ARM=${A%%:*}; R=${A#*:}; RUNG=${R%%:*}; MODE=${R#*:}
    TAG="${OBJ}_${ARM}"
    [ -s "out/TEXEL/${TAG}.json" ] && { SK=$((SK+1)); continue; }
    grep -qx "texdep_${TAG}" /tmp/_de_q.$$ && { SK=$((SK+1)); continue; }
    EXP="OBJ=${OBJ},MODE=${MODE},TAG=${TAG},NFR=150,RUNG=${RUNG}"
    [ -n "${DRY:-}" ] && { echo "DRY texdep_${TAG}  rung${RUNG} mcfm=${MODE}"; N=$((N+1)); continue; }
    J=$(sbatch --parsable --job-name="texdep_${TAG}" --export=ALL,"$EXP" "$E/jobs/texel_after_train.sbatch")
    python3 - "$MAN" "$J" "texdep_${TAG}" "$EXP" "jobs/texel_after_train.sbatch" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
    $JL add "$J" "texdep_${TAG}" "BATCH D+E texel metrics: rung${RUNG} mcfm=${MODE} on ${OBJ}, 150fr | out/texdep_${J}.log | PASS: out/TEXEL/${TAG}.json exists and is non-empty"
    N=$((N+1))
  done
done
rm -f /tmp/_de_q.$$
echo "submitted $N, skipped $SK"
