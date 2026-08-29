#!/bin/bash
# bob_front_go.sh -- orientation solve -> gated 2D copy -> rung27+MCFM, for the yaw-250
# front view of bob (both eyes supervised).
#
# Same three gates as bob_targets_and_train.sh, minus the two-solve agreement check
# (there is only one asset this time): the spot_lava CONTROL must recover identity at
# IoU >= 0.95, the bob row must be verdict PASS, and it must not be flagged AMBIGUOUS --
# bob is a near-symmetric torus and a silently-chosen front/back is how a mesh trains
# back-to-front.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
OBJ=bob_spots_front
ORIENT=$E/out/orient_bob_front/orientation.json
cd "$E"

note() { python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
 $JL add "$1" "$2" "$5"; }

# ---- step 1: orientation solve
J1=$(sbatch --parsable --job-name="orient_${OBJ}" jobs/bob_front_orient.sbatch)
note "$J1" "orient_${OBJ}" "" "jobs/bob_front_orient.sbatch" \
  "solve rotation for ${OBJ} + spot_lava CONTROL; out=out/orient_bob_front/orientation.json; PASS: control IoU>=0.95 and bob PASS not AMBIGUOUS"
echo "  $J1  orient_${OBJ}"

# ---- step 2+3 chained: build_targets_obj then fig27, both afterok
E2="OBJ=${OBJ},ORIENT=${ORIENT}"
J2=$(sbatch --parsable --job-name="mkt_${OBJ}" --dependency=afterok:$J1 --export=ALL,"$E2" jobs/build_targets_obj.sbatch)
note "$J2" "mkt_${OBJ}" "$E2" "jobs/build_targets_obj.sbatch" \
  "bake pose + 2D copy for ${OBJ}; afterok:${J1}; PASS: gt_targets.json exists (log out/mkt_${J2}.log)"
echo "  $J2  mkt_${OBJ}  (afterok:$J1)"

E3="OBJ=${OBJ},MODE=v2_D,NFR=150,MESH=${T2}/data/${OBJ}/mesh/${OBJ}_render_frame.obj,GTDIR=${E}/out/gt_targets_${OBJ}/frames"
J3=$(sbatch --parsable --job-name="27_v2_D_${OBJ}" --dependency=afterok:$J2 --export=ALL,"$E3" jobs/fig27.sbatch)
note "$J3" "27_v2_D_${OBJ}" "$E3" "jobs/fig27.sbatch" \
  "rung27 qkvo+sa r4 + MCFM v2_D, 150fr, ${OBJ}; afterok:${J2}; PASS: '[FINAL] rung' + final_eval.json"
echo "  $J3  27_v2_D_${OBJ}  (afterok:$J2)"
echo "chain: $J1 -> $J2 -> $J3"
