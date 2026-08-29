#!/bin/bash
# bob_resume.sh -- one pass of resume-on-failure for the bob continuation trainings.
#
# WHY THIS EXISTS ALONGSIDE watchdog.py. The watchdog does cover this case on paper --
# TIMEOUT is in BAD, '27_' is in KIND, nightwatch runs -- yet it did not resume
# 2218623 (chair, TIMEOUT at epoch 30/30). Rather than debug that under time pressure,
# these two jobs get their own resume pass keyed on the thing that actually matters:
# whether final_eval.json exists. Job state alone is not the test -- 2218623 reached
# epoch 30 and still had no final_eval, and a run CAN be COMPLETED-but-unusable.
#
# fig27.sbatch resumes from ckpts/, so a resubmit continues rather than restarting.
# MAX_TRIES bounds it so a genuinely broken cell cannot loop forever.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
STATE=$E/out/bob_resume_state.json
MAX_TRIES=${MAX_TRIES:-3}
cd "$E"

for OBJ in bob_spots bob_spots_slow; do
  # 1. finished? final_eval.json is the only real test.
  DONE=$(python3 - "$E" "$OBJ" <<'PY'
import json,sys,glob,os
E,obj=sys.argv[1:3]
for c in glob.glob(f'{E}/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('epochs')!=30 or str(d.get('mcfm'))!='v2_D': continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if os.path.exists(os.path.join(os.path.dirname(c),'final_eval.json')):
        print('done'); break
PY
)
  [ -n "$DONE" ] && { echo "  $OBJ: final_eval.json present — nothing to do"; continue; }

  # 2. already queued or running?
  if squeue -u "$USER" -h -o "%j" | grep -qx "27_v2_D_${OBJ}"; then
    echo "  $OBJ: still live in the queue"; continue
  fi

  # 3. not finished and not live -> it died. Resume.
  N=$(python3 - "$STATE" "$OBJ" <<'PY'
import json,os,sys
p,obj=sys.argv[1:3]
d=json.load(open(p)) if os.path.exists(p) else {}
print(d.get(obj,0))
PY
)
  if [ "$N" -ge "$MAX_TRIES" ]; then
    echo "  $OBJ: $N resumes already, giving up — needs a human"; continue
  fi
  NFR=$( (find "$T2/data/$OBJ/frames_from_video" -maxdepth 1 -name 'frame_*.png' 2>/dev/null || true) | wc -l )
  EXP="OBJ=${OBJ},MODE=v2_D,NFR=${NFR},MESH=${T2}/data/${OBJ}/mesh/${OBJ}_render_frame.obj,GTDIR=${E}/out/gt_targets_${OBJ}/frames"
  J=$(sbatch --parsable --job-name="27_v2_D_${OBJ}" --export=ALL,"$EXP" jobs/fig27.sbatch)
  python3 - "$MAN" "$J" "27_v2_D_${OBJ}" "$EXP" jobs/fig27.sbatch <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  python3 - "$STATE" "$OBJ" <<'PY'
import json,os,sys
p,obj=sys.argv[1:3]
d=json.load(open(p)) if os.path.exists(p) else {}
d[obj]=d.get(obj,0)+1
json.dump(d,open(p,'w'),indent=1)
PY
  $JL add "$J" "27_v2_D_${OBJ}" "RESUME #$((N+1)) of the bob continuation training (no final_eval.json, nothing live); resumes from ckpts"
  echo "  $OBJ: RESUMED as $J (attempt $((N+1))/${MAX_TRIES})"
done
