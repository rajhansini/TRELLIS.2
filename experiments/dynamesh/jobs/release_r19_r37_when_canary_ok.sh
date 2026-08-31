#!/bin/bash
# release_r19_r37_when_canary_ok.sh -- hold the 334-job r19/r37 fan-out until one
# canary of each arm has PROVEN it can draw frames, then submit the rest.
#
# WHY A GATE AT ALL. Neither arm has ever been rendered in pixel space, and the two
# bugs render_arm.py exists to avoid (wrong trainer module, wrong registry shape)
# both surface only at load_state_dict -- after the job starts, before frame 1. A
# blind fan-out would put 334 identical failures in the queue. render_arm.py's own
# header records that both bugs were found by exactly one canary job.
#
# PASS CONDITION is frames on disk, not exit status: the renderer writes
# out/<tag>/frames/0001.png only after the registry loaded, the checkpoint matched
# and the first ODE decode finished. Five frames is past every failure mode above.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
LOGD=$E/out/RENDERS
OBJ=${OBJ:-ancient_lady_effect_2}
NEED=${NEED:-5}
cd "$E"

frames() {  # tag -> png count, 0 if the dir is absent
  local d="$E/out/view_${OBJ}_$1_train/frames"
  [ -d "$d" ] && find "$d" -maxdepth 1 -name '*.png' | wc -l || echo 0
}
dead() {    # jobname -> non-empty if a terminal-bad state or a python traceback
  # `|| true` is not decoration. Under `set -eo pipefail` a grep that matches
  # nothing exits 1, pipefail propagates it, and `D19=$(dead ...)` then kills the
  # whole gate through set -e -- which is exactly how the first launch died two
  # seconds after arming, before either canary had even written a log.
  local jn=$1
  { grep -lE "Traceback|Unexpected key|Missing key|FAILED:|CUDA error|Error executing" \
      "$LOGD/${jn}_"*.log 2>/dev/null || true; } | head -1
}

echo "gate armed $(date '+%F %T')  obj=$OBJ  need=${NEED} frames per arm"
while :; do
  F19=$(frames 19); F37=$(frames 37)
  D19=$(dead "r19v_${OBJ}_train"); D37=$(dead "r37v_${OBJ}_train")
  echo "$(date '+%T')  r19=${F19}fr  r37=${F37}fr  ${D19:+r19-ERR }${D37:+r37-ERR}"
  if [ -n "$D19" ] || [ -n "$D37" ]; then
    echo "GATE FAILED: canary error"
    [ -n "$D19" ] && { echo "--- r19 ---"; tail -25 "$D19"; }
    [ -n "$D37" ] && { echo "--- r37 ---"; tail -25 "$D37"; }
    echo "NOT releasing the fan-out. Fix, then rerun this script."
    exit 1
  fi
  [ "$F19" -ge "$NEED" ] && [ "$F37" -ge "$NEED" ] && break
  sleep 60
done

echo "GATE PASSED $(date '+%F %T'): both canaries are drawing frames. Releasing."
ARM="r19 r37" bash "$E/jobs/submit_arm_renders.sh"
echo "release complete $(date '+%F %T')"
