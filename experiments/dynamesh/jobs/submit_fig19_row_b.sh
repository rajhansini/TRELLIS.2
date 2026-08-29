#!/bin/bash
# submit_fig19_row_b.sh — row (b) of the ablation table: cross-attention-only LoRA.
#
# One job per object, never a loop inside one job. Params come from each object's
# OWN rung27 config so the CA-only arm is matched to its CA+SA partner on mesh,
# targets dir and frame count -- hardcoding hero paths would train the guan-pipeline
# object (pumpkin_rot, 121 frames) against the wrong targets and still exit 0.
#
# pumpkin_rot and spot_lava are SKIPPED: they already have complete rung19 runs
# (3bb4375e, 60c8897c) with configs matched to their rung27 partners.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
PARAMS=$E/out/fig19_params.tsv
mkdir -p $E/out/FIGRUNS
[ -f "$PARAMS" ] || { echo "missing $PARAMS"; exit 1; }

n=0
while IFS=$'\t' read -r OBJ NFR MESH GTDIR; do
  [ "$OBJ" = "object" ] && continue
  [ -z "$OBJ" ] && continue
  JN="19_${OBJ}"
  EXP="OBJ=${OBJ},NFR=${NFR},MESH=${MESH},GTDIR=${GTDIR}"
  JID=$(sbatch --parsable --job-name="$JN" --export=ALL,"$EXP" "$E/jobs/fig19.sbatch")
  python3 - "$MAN" "$JID" "$JN" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/fig19.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  /net/projects/ranalab/rajhansini/joblog.sh add "$JID" "fig19_${OBJ}" \
    "rung19 CA-only LoRA (--targets qkvo), ${NFR}fr, row(b) ablation; out=$E/out/FIGRUNS/${JN}_${JID}.log; PASS: log has '[FINAL] rung' and runs/rung19_l1_lp_all_qkvo_r4_s42_*/final_eval.json exists"
  echo "$JID  $JN  nfr=$NFR"
  n=$((n+1))
done < "$PARAMS"
echo "submitted $n jobs"
