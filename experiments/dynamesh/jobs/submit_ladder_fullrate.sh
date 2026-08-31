#!/bin/bash
# submit_ladder_fullrate.sh -- full-rate (150-frame) VIDEO-SPACE metrics for the
# ablation-ladder arms r19 / r37 / r31 / r31m, now that all 376 renders exist.
#
# WHAT THIS PRODUCES. The pixel-space twin of the texel component ladder: flicker,
# accel, drift plus PSNR/SSIM for each arm at the training view and the three unseen
# diagonals. Same driver, same masks, same panel crops and same 2D-copy reference as
# the window arms measured by submit_window_fullrate.sh, so the ladder rows and the
# window rows can sit in one table. Using a different reference is how out/FULLRATE_W11
# ended up reading ~6 dB low.
#
# THE FROZEN PANEL IS ALWAYS view_<obj>_27_<view>, not this arm's own left half.
# fullrate_metrics_arm.py takes it from the non-MCFM run deliberately: inside an
# --mcfm run the frozen arm receives the blended conditioning too, so that left half
# is frozen+MCFM rather than frozen. Do not "fix" this by reading the local panel.
#
# CPU ONLY, no --gres. The cost is ~2,550 NFS reads per object against ~0 compute, so
# a GPU would idle while holding a slot the W=13/15 training fleet needs.
#
# One job per (arm, object), idempotent on the output json.
#   ARMS="19 37 31 31m"   restrict the arms
#   ONLY=obj1,obj2        restrict the objects
#   DRY=1                 print the work list, submit nothing
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
JL=/net/projects/ranalab/rajhansini/joblog.sh
MAN=$E/out/job_manifest.json
cd "$E"
OBJS=$(python3 -c "import json;print(' '.join(json.load(open('out/fullrate_table_batchALL.json'))['objects']))")
squeue -u "$USER" -h -o "%j" > /tmp/_lf.$$ || true

N=0; S=0; SKIP=0; NOREND=""
# arm 27 = rung27 with MCFM OFF, the "+self-attention, temporal off" ladder row.
# fullrate_metrics_arm.py already reads the frozen panel from view_<obj>_27_<view>,
# so for this arm both panels come from one render: left frozen, right ours.
for SPEC in "19:FULLRATE_R19:fr19" "27:FULLRATE_R27:fr27" "37:FULLRATE_R37:fr37" "31:FULLRATE_R31:fr31" "31m:FULLRATE_R31M:fr31m"; do
  ARM=${SPEC%%:*}; R=${SPEC#*:}; OUTDIR=${R%%:*}; PFX=${R##*:}
  [ -n "${ARMS:-}" ] && [[ " $ARMS " != *" $ARM "* ]] && continue
  mkdir -p "out/$OUTDIR"
  for OBJ in $OBJS; do
    [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
    N=$((N+1))
    [ -s "out/$OUTDIR/$OBJ.json" ] && { SKIP=$((SKIP+1)); continue; }
    grep -qx "${PFX}_${OBJ}" /tmp/_lf.$$ && { SKIP=$((SKIP+1)); continue; }
    # Every camera must be rendered or the unseen-view average is taken over a
    # different set of views than the other arms. NEVER `ls glob | wc -l` here:
    # under `set -eo pipefail` a missing dir makes ls exit 2 and kills the script
    # silently, which is exactly how submit_ladder_renders.sh truncated itself.
    OK=1
    for V in train diagA diagB diagC; do
      C=0
      [ -d "out/view_${OBJ}_${ARM}_${V}/frames" ] && \
        C=$(find "out/view_${OBJ}_${ARM}_${V}/frames" -maxdepth 1 -name '*.png' | wc -l)
      [ "$C" -ge 121 ] || OK=0
    done
    # r31/r31m exist for 24 of 42 objects; the other 18 were never trained at rung31,
    # so they are reported here rather than submitted against absent frames.
    [ "$OK" -eq 0 ] && { NOREND="$NOREND ${ARM}/${OBJ}"; continue; }
    [ -n "${DRY:-}" ] && { echo "   $OBJ arm=$ARM -> $OUTDIR"; continue; }
    EXP="OBJ=${OBJ},ARM=${ARM},OUTDIR=${OUTDIR}"
    J=$(sbatch --parsable --job-name="${PFX}_${OBJ}" --export=ALL,"$EXP" jobs/fullrate_obj_arm.sbatch)
    python3 - "$MAN" "$J" "${PFX}_${OBJ}" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/fullrate_obj_arm.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
    $JL add "$J" "${PFX}_${OBJ}" \
      "VIDEO-SPACE ladder metrics: ${OBJ} arm=${ARM}, 150fr at 4 cameras, flicker/accel/drift + PSNR/SSIM vs the 2D copy -> out/${OUTDIR}/${OBJ}.json; log out/fr_${J}.log; PASS: json non-empty"
    echo "  $J  ${PFX}_${OBJ}"
    S=$((S+1))
  done
done
rm -f /tmp/_lf.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
[ -n "$NOREND" ] && { echo "NO RENDERS (never trained at that rung, not a failure):"; \
  for x in $NOREND; do echo "   $x"; done; }
exit 0
