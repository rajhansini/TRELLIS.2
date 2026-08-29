#!/bin/bash
# submit_window_fullrate.sh -- full-rate pixel metrics for the MCFM window arms.
#
# One job per (arm, object). Idempotent on the output json, so a rerun submits
# only what is missing. All three arms are measured with fullrate_metrics_arm.py,
# which scores PSNR/SSIM against the 2D COPY (out/gt_targets_<obj>/frames), the
# same reference the 42-object comparison table uses. The pre-existing
# out/FULLRATE_W11/ was produced by fullrate_metrics_w11.py against the RAW video
# frame and reads ~6 dB lower, so W=11 is re-measured here rather than reused --
# four rows on one reference or the rows are not comparable.
#
# CPU ONLY. These jobs take no GPU; they must not compete with the render fleet.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
JL=/net/projects/ranalab/rajhansini/joblog.sh
MAN=$E/out/job_manifest.json
cd "$E"
OBJS=$(python3 -c "import json;print(' '.join(json.load(open('out/fullrate_table_batchALL.json'))['objects']))")
squeue -u "$USER" -h -o "%j" > /tmp/_fw.$$ || true
N=0; S=0; SKIP=0; NOREND=0
for SPEC in "27e:FULLRATE_W5:fw5" "27f:FULLRATE_W7:fw7" "27g:FULLRATE_W11CG:fw11"; do
  ARM=${SPEC%%:*}; R=${SPEC#*:}; OUTDIR=${R%%:*}; PFX=${R##*:}
  mkdir -p "out/$OUTDIR"
  for OBJ in $OBJS; do
    N=$((N+1))
    [ -s "out/$OUTDIR/$OBJ.json" ] && { SKIP=$((SKIP+1)); continue; }
    grep -qx "${PFX}_${OBJ}" /tmp/_fw.$$ && { SKIP=$((SKIP+1)); continue; }
    # every camera must be rendered or the unseen average is wrong
    OK=1
    for V in train diagA diagB diagC; do
      C=$(ls "out/view_${OBJ}_${ARM}_${V}/frames"/*.png 2>/dev/null | wc -l)
      [ "$C" -ge 121 ] || OK=0
    done
    [ "$OK" -eq 0 ] && { NOREND=$((NOREND+1)); continue; }
    [ -n "${DRY:-}" ] && { echo "   $OBJ $ARM -> $OUTDIR"; continue; }
    EXP="OBJ=${OBJ},ARM=${ARM},OUTDIR=${OUTDIR}"
    J=$(sbatch --parsable --job-name="${PFX}_${OBJ}" --export=ALL,"$EXP" jobs/fullrate_obj_arm.sbatch)
    python3 - "$MAN" "$J" "${PFX}_${OBJ}" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/fullrate_obj_arm.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
    $JL add "$J" "${PFX}_${OBJ}" "window fullrate: ${OBJ} arm=${ARM} -> out/${OUTDIR}/${OBJ}.json; PASS: non-empty json"
    S=$((S+1))
  done
done
rm -f /tmp/_fw.$$
echo "cells ${N}; submitted ${S}; already done/queued ${SKIP}; renders incomplete ${NOREND}"
