#!/bin/bash
# submit_dailies_de.sh -- daily render panels for batches D and E, matching what
# batch C already has: view_<obj>_27_<view> and view_<obj>_27m_<view> at the training
# view and the three unseen diagonals. Each panel is frozen | adapted side by side, so
# the two arms together give the four-column GT / frozen / rung27 / rung27+MCFM daily.
#
# RUN DIRS ARE RESOLVED FROM config.json, never globbed by name: the run hash is not
# derivable from the object, and substring matching on gt_dir is a trap ('dragon_mush'
# is a prefix of 'dragon_mush2'). Exact '/data/<obj>/' segment match only.
# DRY=1 prints the work list. NICE deprioritises against other work.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="moai_silver moai_animated gargoyle_spiral gargoyle_effect_one airplane_blub airplane_red_cracks teddy_bleach teddy_fusion goat_burnt goat_flower goat_clay ivysaur_petal ivysaur_petal_2 blub_raurshaw blub_drying pegasus_shine pegaso_effect_1 mosaic_painting"

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

resolve() {  # obj mcfm(None|v2_D) -> run dir, or empty
  python3 - "$E" "$1" "$2" <<'PY'
import json,sys,glob,os
E,obj,mode=sys.argv[1:4]
hit=[]
for c in glob.glob(f'{E}/runs/rung27_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if str(d.get('mcfm'))!=mode: continue
    if d.get('rung')!=27 or d.get('epochs')!=30: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if not os.path.exists(os.path.join(os.path.dirname(c),'ckpts','lora_best.pt')): continue
    hit.append(os.path.dirname(c))
print(hit[0] if len(hit)==1 else '')
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_dl.$$ || true
N=0; S=0; MISS=""
for OBJ in $OBJS; do
  for SPEC in "27:None" "27m:v2_D"; do
    TAG=${SPEC%%:*}; MODE=${SPEC##*:}
    RUN=$(resolve "$OBJ" "$MODE")
    [ -z "$RUN" ] && { MISS="$MISS ${OBJ}/${TAG}"; continue; }
    for V in "train:0.0:0.0" "diagA:45.0:25.0" "diagB:135.0:-20.0" "diagC:225.0:30.0"; do
      VIEW=${V%%:*}; R=${V#*:}; YAW=${R%%:*}; ELEV=${R##*:}
      N=$((N+1))
      OUTD="$E/out/view_${OBJ}_${TAG}_${VIEW}"
      [ -d "$OUTD/frames" ] && [ "$(ls "$OUTD/frames"/*.png 2>/dev/null | wc -l)" -ge 150 ] && continue
      grep -qx "dl_${OBJ}_${TAG}_${VIEW}" /tmp/_dl.$$ && continue
      if [ -n "${DRY:-}" ]; then printf '   %-24s %-4s %-6s yaw=%-6s elev=%s\n' "$OBJ" "$TAG" "$VIEW" "$YAW" "$ELEV"; continue; fi
      EXP="RUN=${RUN},TAG=view_${OBJ}_${TAG}_${VIEW},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=150,RES=518"
      J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="dl_${OBJ}_${TAG}_${VIEW}" \
           --export=ALL,"$EXP" "$E/jobs/render_arm.sbatch")
      note "$J" "dl_${OBJ}_${TAG}_${VIEW}" "$EXP" "jobs/render_arm.sbatch" \
        "DAILIES batch D/E: ${OBJ} rung27${MODE:+ mcfm=$MODE} at ${VIEW} (yaw ${YAW}, elev ${ELEV}), 150fr, frozen|adapted panel -> out/view_${OBJ}_${TAG}_${VIEW}/; PASS: 150 pngs in frames/"
      echo "  $J  dl_${OBJ}_${TAG}_${VIEW}"
      S=$((S+1))
    done
  done
done
rm -f /tmp/_dl.$$
echo; echo "cells considered ${N}; submitted ${S}"
[ -n "$MISS" ] && echo "UNRESOLVED (no unique 30-epoch run with a checkpoint):$MISS"
