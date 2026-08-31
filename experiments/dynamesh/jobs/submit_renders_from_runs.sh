#!/bin/bash
# submit_renders_from_runs.sh -- pixel renders (train + 3 unseen diagonals) for objects
# that are TRAINED but have no texel pass yet.
#
# WHY NOT submit_batchfg_renders.sh. That script resolves each (object, arm) run from
# out/TEXEL/<obj>_<arm>.json, so the pixel row and the texel row are guaranteed to be the
# same checkpoint. car_effect1/2 and robot_rust_2 finished training this morning but have
# no texel pass, so that lookup returns nothing and the renders can never be submitted.
# Here the run is resolved from runs/*/config.json by exact '/data/<obj>/' match plus
# (rung, mcfm), which is the same rule submit_dailies_de.sh uses. When the texel pass
# later runs it will pick the same dir, because the (object, rung, mcfm, epochs) key is
# unique -- asserted below, not assumed: more than one match is a hard error.
#
#   OBJS=...   objects (default: the three batch-G clips that need this)
#   ARMS=...   comma list (default all seven)
#   DRY=1      print the work list, submit nothing
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="${OBJS:-car_effect1 car_effect2 robot_rust_2}"
ARMS="${ARMS:-r19,r19mcfm,r27,r27mcfm,w5,stD,v3D}"

N=0; S=0; SKIP=0; NOCK=""
squeue -u "$USER" -h -o "%j" > /tmp/_rr.$$ || true
for OBJ in $OBJS; do
  echo "== $OBJ"
  for SPEC in "r19:19:None" "r19mcfm:19:v2_D" "r27:27:None" "r27mcfm:27:v2_D" \
              "w5:27:v2_E" "stD:27:st_D" "v3D:27:v3_D"; do
    ARM=${SPEC%%:*}; R=${SPEC#*:}; RUNG=${R%%:*}; MODE=${R##*:}
    [[ ",$ARMS," != *",$ARM,"* ]] && continue
    RUN=$(python3 - "$E" "$OBJ" "$RUNG" "$MODE" <<'PY'
import json,sys,glob,os
E,obj,rung,mode=sys.argv[1:5]
hit=[]
for c in glob.glob(f'{E}/runs/rung{rung}_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('rung')!=int(rung) or str(d.get('mcfm'))!=mode: continue
    if d.get('epochs')!=30: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    dd=os.path.dirname(c)
    if not os.path.exists(f'{dd}/ckpts/lora_best.pt'): continue
    hit.append(dd)
# more than one match means the key is not unique and the pixel row could disagree
# with the texel row later -- refuse rather than pick one.
assert len(hit)<2, f'{obj}/{rung}/{mode}: {len(hit)} matching runs {hit}'
print(hit[0] if hit else '')
PY
)
    if [ -z "$RUN" ]; then NOCK="$NOCK ${OBJ}/${ARM}"; echo "   no checkpoint yet: $ARM"; continue; fi
    for V in "train:0.0:0.0" "diagA:45.0:25.0" "diagB:135.0:-20.0" "diagC:225.0:30.0"; do
      VIEW=${V%%:*}; R2=${V#*:}; YAW=${R2%%:*}; ELEV=${R2##*:}
      TAG="view_${OBJ}_${ARM}_${VIEW}"
      JN="rfg_${OBJ}_${ARM}_${VIEW}"
      N=$((N+1))
      if [ -d "out/${TAG}/frames" ]; then
        C=$(find "out/${TAG}/frames" -maxdepth 1 -name '*.png' | wc -l)
        [ "$C" -ge 150 ] && { SKIP=$((SKIP+1)); continue; }
      fi
      grep -qx "$JN" /tmp/_rr.$$ && { SKIP=$((SKIP+1)); continue; }
      [ -n "${DRY:-}" ] && { printf '   %-9s %-6s %s\n' "$ARM" "$VIEW" "${RUN##*/}"; continue; }
      EXP="RUN=${RUN},TAG=${TAG},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=150,RES=518"
      J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" --export=ALL,"$EXP" jobs/render_arm.sbatch)
      python3 - "$MAN" "$J" "$JN" "$EXP" <<'PY'
import json,os,sys
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/render_arm.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
      $JL add "$J" "$JN" "batch G render: ${OBJ} arm=${ARM} view=${VIEW} (${YAW},${ELEV}) 150fr -> out/${TAG}/frames; run=${RUN##*/}; PASS: 150 pngs"
      echo "   $J  $ARM  $VIEW"
      S=$((S+1))
    done
  done
done
rm -f /tmp/_rr.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
[ -n "$NOCK" ] && echo "NO CHECKPOINT (not submitted):$NOCK"
exit 0
