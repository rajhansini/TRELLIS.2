#!/bin/bash
# Pixel-space renders for the batch F and G objects: one SEEN camera (train) and
# three UNSEEN (diagA/B/C), same four the published pixel table uses.
#
# The run for each (object, arm) is read from the SAME out/TEXEL/<obj>_<arm>.json
# the texel numbers came from, exactly as submit_ladder_renders.sh does, so the
# pixel row and the texel row are the same checkpoint rather than two runs that
# merely share a config hash.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
JL=/net/projects/ranalab/rajhansini/joblog.sh
MAN=$E/out/job_manifest.json
cd $E
OBJS=${OBJS:-"doorknob_spinodal doorknob_bz bunny_shine unicorn_rainbow unicorn_effect1"}
ARMS=${ARMS:-"r19 r27 r27mcfm"}
N=0; SKIP=0; MISS=""
for OBJ in $OBJS; do
  for ARM in $ARMS; do
    RUN=$(python3 - "$E" "$OBJ" "$ARM" <<'PY'
import json, os, sys
E, obj, arm = sys.argv[1:4]
p = f'{E}/out/TEXEL/{obj}_{arm}.json'
if not os.path.exists(p): print(''); raise SystemExit
d = f"{E}/runs/{json.load(open(p))['run'].split('/')[-1]}"
print(d if os.path.exists(f'{d}/ckpts/lora_best.pt') and os.path.exists(f'{d}/config.json') else '')
PY
)
    [ -z "$RUN" ] && { MISS="$MISS ${OBJ}/${ARM}"; continue; }
    for V in "train:0.0:0.0" "diagA:45.0:25.0" "diagB:135.0:-20.0" "diagC:225.0:30.0"; do
      VIEW=${V%%:*}; R=${V#*:}; YAW=${R%%:*}; ELEV=${R##*:}
      TAG="view_${OBJ}_${ARM}_${VIEW}"
      if [ -d "out/${TAG}/frames" ]; then
        C=$(find "out/${TAG}/frames" -maxdepth 1 -name '*.png' | wc -l)
        [ "$C" -ge 150 ] && { SKIP=$((SKIP+1)); continue; }
      fi
      EXP="RUN=${RUN},TAG=${TAG},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=150,RES=518"
      J=$(sbatch --parsable --job-name="rfg_${OBJ}_${ARM}_${VIEW}" --export=ALL,"$EXP" jobs/render_arm.sbatch)
      python3 - "$MAN" "$J" "rfg_${OBJ}_${ARM}_${VIEW}" "$EXP" <<'PY'
import json, os, sys
man, jid, name, exp = sys.argv[1:5]
m = json.load(open(man)) if os.path.exists(man) else {}
m[jid] = {"script": "jobs/render_arm.sbatch", "name": name, "export": exp}
json.dump(m, open(man, "w"), indent=1)
PY
      $JL add "$J" "rfg_${OBJ}_${ARM}_${VIEW}" \
        "batch F/G render: ${OBJ} arm=${ARM} view=${VIEW} (${YAW},${ELEV}) 150fr -> out/${TAG}/frames; PASS: 150 pngs"
      N=$((N+1)); echo "  $J  ${OBJ}  ${ARM}  ${VIEW}"
    done
  done
done
echo "submitted $N, skipped $SKIP already rendered"
[ -n "$MISS" ] && { echo "NO TEXEL JSON / NO CHECKPOINT:"; for x in $MISS; do echo "   $x"; done; }
