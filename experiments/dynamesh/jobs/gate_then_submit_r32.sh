#!/bin/bash
# Wait on the rung32 render smoke test, and submit the 16 rung32 panel jobs ONLY
# if it printed the GATE-window PASS line.
#
# The gate matters more than usual here. A cw3 checkpoint has the same 300 keys as
# a plain rung27 one, so if the context-window patch is wrong the render does not
# crash -- it produces plausible frames conditioned on a third of the context the
# adapter was trained on, and the error is invisible until someone diffs it against
# training. So: no PASS line, no jobs.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
SMOKE=2192789
LOG=$E/out/rview_${SMOKE}.log
OUT=$E/out/gate_r32.log
exec >>"$OUT" 2>&1
echo "[$(date +%H:%M:%S)] waiting on smoke $SMOKE"
for i in $(seq 1 360); do          # up to 3 h
  if [ -f "$LOG" ] && grep -q "GATE-window PASSED" "$LOG"; then
    echo "[$(date +%H:%M:%S)] GATE PASSED"
    grep -E "^\[R32\]" "$LOG" | sed 's/^/    /'
    n=0
    while IFS=$'\t' read -r O ARM RUN NFR; do
      [ "$ARM" = "rung32" ] || continue
      for Y in 0 90 180 270; do
        TAG="panel_${O}_${ARM}_y${Y}"
        J=$(sbatch --parsable --job-name=pv_rung32 \
            --export=ALL,RUN=$RUN,TAG=$TAG,NFR=$NFR,YAW0=$Y,ELEV=15,TURNS=0,RES=518,ARM=$ARM \
            $E/jobs/panel_view.sbatch)
        echo -e "$O\trung32\t$Y\t$J\t$TAG" >> $E/out/panel_jobs.tsv
        /net/projects/ranalab/rajhansini/joblog.sh add "$J" "pview_${O}_rung32_y${Y}" \
          "panel render ${O} rung32(cw3) yaw${Y} fixed-cam 518px | PASS: out/${TAG}/frames full count" >/dev/null
        n=$((n+1))
      done
    done < /tmp/arms4.tsv
    echo "[$(date +%H:%M:%S)] submitted $n rung32 panel jobs (expect 16)"
    exit 0
  fi
  if [ -f "$LOG" ] && grep -qE "GATE-window FAILED|Traceback|AssertionError" "$LOG"; then
    echo "[$(date +%H:%M:%S)] GATE FAILED — NOT submitting rung32. Offending lines:"
    grep -E -A4 "GATE-window FAILED|Traceback|AssertionError" "$LOG" | head -25
    exit 1
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] timed out waiting for smoke $SMOKE — nothing submitted"
exit 2
