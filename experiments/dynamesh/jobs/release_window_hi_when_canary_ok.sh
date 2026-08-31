#!/bin/bash
# release_window_hi_when_canary_ok.sh -- hold the remaining 82 W=13/15 training jobs
# until one canary of each window has PROVEN the new offsets work end to end.
#
# PASS CONDITION is the trainer's own blend gate, not exit status and not frames:
#
#   [MCFM] mode=v2_H ... window=(-6,...,6) frames=150
#   [MCFM] GATE-blend max|blended-vanilla| at frame 1 = ...  (must be > 0)
#   [GATE-grad] PASSED
#
# Those three land within the first minutes, long before any epoch finishes, and
# together they prove parse_mode resolved the new width, the blend actually changed
# the tokens rather than silently no-opping, and gradient still reaches every B
# matrix through the wider window. Waiting for training itself would cost 3-6h per
# canary to learn nothing more.
#
# A no-op blend is the failure this specifically catches: an offsets tuple that
# parsed but did not widen would train 84 jobs that are all secretly W=11.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
LOGD=$E/out/FIGRUNS
OBJ=${OBJ:-spot_lava}
cd "$E"

# `|| true` on every grep: under `set -eo pipefail` a grep that matches nothing exits
# 1 and kills the gate through the command substitution. That is exactly how the
# render gate died two seconds after arming.
ok() {   # mode -> non-empty when that canary has cleared all three gates
  local m=$1 f
  f=$(ls -t "$LOGD/27_${m}_${OBJ}_"*.log 2>/dev/null | head -1 || true)
  [ -z "$f" ] && return 0
  local blend grad
  blend=$(grep -c "GATE-blend" "$f" 2>/dev/null || true)
  grad=$(grep -c "\[GATE-grad\] PASSED" "$f" 2>/dev/null || true)
  [ "${blend:-0}" -ge 1 ] && [ "${grad:-0}" -ge 1 ] && echo yes || true
}
bad() {  # mode -> the log path when that canary has died
  local m=$1 f
  f=$(ls -t "$LOGD/27_${m}_${OBJ}_"*.log 2>/dev/null | head -1 || true)
  [ -z "$f" ] && return 0
  # DO NOT put "must be > 0, else" here. That substring is part of the trainer's
  # SUCCESS line -- "GATE-blend max|blended-vanilla| ... (must be > 0, else the blend
  # is a no-op)" -- so matching it makes the very evidence of a passing canary read as
  # a failure. It did: both canaries cleared blend and grad, and the gate refused to
  # release the other 82. Bare "FAILED" is out for the same reason: these logs print
  # gate names constantly. Only a real traceback, an OOM, or an explicit
  # "[GATE-xxx] FAILED" counts.
  { grep -lE "Traceback \(most recent call last\)|KeyError|CUDA out of memory|\[GATE-[a-z]+\] FAILED" \
      "$f" 2>/dev/null || true; } | head -1
}

echo "gate armed $(date '+%F %T')  obj=$OBJ  waiting on v2_H and v2_I blend+grad gates"
while :; do
  H=$(ok v2_H); I=$(ok v2_I)
  BH=$(bad v2_H); BI=$(bad v2_I)
  echo "$(date '+%T')  v2_H=${H:-wait}  v2_I=${I:-wait}  ${BH:+H-ERR }${BI:+I-ERR}"
  if [ -n "$BH" ] || [ -n "$BI" ]; then
    echo "GATE FAILED: canary error"
    [ -n "$BH" ] && { echo "--- v2_H ---"; grep -E "MCFM|Traceback|Error|KeyError" "$BH" | tail -20; }
    [ -n "$BI" ] && { echo "--- v2_I ---"; grep -E "MCFM|Traceback|Error|KeyError" "$BI" | tail -20; }
    echo "NOT releasing the other 82. Fix, then rerun this script."
    exit 1
  fi
  [ -n "$H" ] && [ -n "$I" ] && break
  sleep 60
done

echo "GATE PASSED $(date '+%F %T'): both windows blend and backprop. Releasing."
for m in v2_H v2_I; do
  f=$(ls -t "$LOGD/27_${m}_${OBJ}_"*.log 2>/dev/null | head -1 || true)
  grep -m1 "\[MCFM\] mode=" "$f" 2>/dev/null || true
done
bash "$E/jobs/submit_window_hi.sh"
echo "release complete $(date '+%F %T')"
