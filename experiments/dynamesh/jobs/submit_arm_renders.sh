#!/bin/bash
# submit_arm_renders.sh -- pixel (video-space) renders for the r19 / r37 / r31 arms.
#
# WHY THIS EXISTS
#   The texel-space tables are complete; their video-space twins are not, because
#   these three arms were never rendered at the four fixed cameras that every other
#   arm has. out/view_<obj>_27_<view> and _27g_ exist for all 42 objects, so once
#   these land the component ladder (frozen -> r19 -> r27 -> w11) and the temporal
#   rows (r31, r37) can be recomputed in pixels over the SAME object set as the
#   texel version, which is the whole point of the row matching the paper.
#
# ONE JOB PER (object, view). Never a loop inside one job.
#
# RUN DIRS ARE RESOLVED FROM config.json, never globbed by name: the run hash is not
# derivable from the object, and substring matching on gt_dir is a trap ('dragon_mush'
# is a prefix of 'dragon_mush2'). Exact '/data/<obj>/' segment match only, and a cell
# is skipped unless the match is UNIQUE.
#
# THE OBJECT SET IS PER-ARM, NOT 42 ACROSS THE BOARD.
#   r19 and r37 have 30-epoch checkpoints for all 42 objects.
#   r31 and r31m have them for 24 (batches A+B). The 18 batch-D/E objects were never
#   trained at rung31, so asking for 42 there would submit 72 jobs that die on a
#   missing run dir. Measured, not assumed -- see the UNRESOLVED report at the end.
#
# CAMERA. turns=0, so the camera is FIXED for the whole sequence and every change
# between frames is texture. Same four cameras as the 27/27g renders: the training
# view plus three unseen diagonals.
#
# IDEMPOTENT. A cell with NFR pngs already on disk is skipped; so is one whose job
# name is still in the queue. Re-running this script IS the resubmit mechanism.
#
#   ARM=r19|r37|r31|r31m   which arm (default: all four)
#   ONLY=obj1,obj2         restrict to named objects
#   LIMIT=n                submit at most n jobs (canary runs)
#   DRY=1                  print the work list, submit nothing
#   NICE=n                 deprioritise against other work
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
cd "$E"
[ -s "$TSV" ] || { echo "missing $TSV"; exit 1; }
mkdir -p "$E/out/RENDERS"

# arm : run-dir prefix : required mcfm : required rung : out-tag
ARMS_ALL="r19 r37 r31 r31m"
spec_prefix() { case $1 in
  r19)  echo 'rung19_l1_lp_all_qkvo_r4_s42' ;;
  r27)  echo 'rung27_l1_lp_all_qkvo+sa_r4_s42' ;;   # MCFM off; resolve() keeps only epochs==30, which is unique
  r37)  echo 'rung37_l1_lp_tw3bpp_all_qkvo+sa_r4_s42' ;;
  r31)  echo 'rung31_l1_lp_tw3b_all_qkvo+sa_r4_s42' ;;
  r31m) echo 'rung31_l1_lp_mcfmv2_D_tw3b_all_qkvo+sa_r4_s42' ;;
esac; }
spec_mcfm() { case $1 in r31m) echo 'v2_D' ;; *) echo 'None' ;; esac; }
spec_rung() { case $1 in r19) echo 19 ;; r27) echo 27 ;; r37) echo 37 ;; *) echo 31 ;; esac; }
spec_tag()  { case $1 in r19) echo 19 ;; r27) echo 27 ;; r37) echo 37 ;; r31) echo 31 ;; r31m) echo 31m ;; esac; }

resolve() {  # prefix mcfm rung obj -> run dir, or empty if absent/ambiguous
  python3 - "$E" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,glob,os
E,pref,mode,rung,obj=sys.argv[1:6]
hit=[]
for c in glob.glob(f'{E}/runs/{pref}*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if str(d.get('mcfm'))!=mode: continue
    if d.get('rung')!=int(rung) or d.get('epochs')!=30: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if not os.path.exists(os.path.join(os.path.dirname(c),'ckpts','lora_best.pt')): continue
    hit.append(os.path.dirname(c))
print(hit[0] if len(hit)==1 else '')
PY
}

note() {
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
}

squeue -u "$USER" -h -o "%j" > /tmp/_arm.$$ || true
N=0; S=0; SKIP=0; MISS=""
for ARM in ${ARM:-$ARMS_ALL}; do
  PREF=$(spec_prefix "$ARM"); MODE=$(spec_mcfm "$ARM")
  RUNG=$(spec_rung "$ARM");   TAG=$(spec_tag "$ARM")
  [ -z "$PREF" ] && { echo "unknown ARM '$ARM'"; exit 1; }
  while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
    [ "$B" = "batch" ] && continue
    [ -z "$OBJ" ] && continue
    [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
    NFR=${NFR:-150}
    RUN=$(resolve "$PREF" "$MODE" "$RUNG" "$OBJ")
    [ -z "$RUN" ] && { MISS="$MISS ${ARM}/${OBJ}"; continue; }
    for V in "train:0.0:0.0" "diagA:45.0:25.0" "diagB:135.0:-20.0" "diagC:225.0:30.0"; do
      VIEW=${V%%:*}; R=${V#*:}; YAW=${R%%:*}; ELEV=${R##*:}
      N=$((N+1))
      OUTD="$E/out/view_${OBJ}_${TAG}_${VIEW}"
      # never `ls glob | wc -l` under pipefail: a missing dir kills the script
      HAVE=0
      [ -d "$OUTD/frames" ] && HAVE=$(find "$OUTD/frames" -maxdepth 1 -name '*.png' | wc -l)
      [ "$HAVE" -ge "$NFR" ] && { SKIP=$((SKIP+1)); continue; }
      JN="${ARM}v_${OBJ}_${VIEW}"
      grep -qx "$JN" /tmp/_arm.$$ && { SKIP=$((SKIP+1)); continue; }
      if [ -n "${DRY:-}" ]; then
        printf '   %-5s %-24s %-6s yaw=%-6s elev=%-6s nfr=%-4s have=%s\n' \
          "$ARM" "$OBJ" "$VIEW" "$YAW" "$ELEV" "$NFR" "$HAVE"; continue
      fi
      [ -n "${LIMIT:-}" ] && [ "$S" -ge "$LIMIT" ] && continue
      EXP="RUN=${RUN},TAG=view_${OBJ}_${TAG}_${VIEW},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=${NFR},RES=518"
      J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" \
           --export=ALL,"$EXP" "$E/jobs/render_arm.sbatch")
      note "$J" "$JN" "$EXP" "jobs/render_arm.sbatch" \
        "VIDEO-SPACE ${ARM}: ${OBJ} at ${VIEW} (yaw ${YAW}, elev ${ELEV}), ${NFR}fr fixed camera, frozen|adapted panel -> out/view_${OBJ}_${TAG}_${VIEW}/; log out/RENDERS/${JN}_*.log; PASS: ${NFR} pngs in frames/"
      echo "  $J  $JN"
      S=$((S+1))
    done
  done < "$TSV"
done
rm -f /tmp/_arm.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
[ -n "$MISS" ] && { echo "UNRESOLVED (no unique 30-epoch run with a checkpoint):"; \
  for m in $MISS; do echo "   $m"; done; }
exit 0
