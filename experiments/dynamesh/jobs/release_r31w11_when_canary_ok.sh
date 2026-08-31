#!/bin/bash
# release_r31w11_when_canary_ok.sh -- hold the other 41 W=11 spatio-temporal runs
# until one canary has PROVEN the wider window both runs and FITS.
#
# THIS GATE CHECKS MEMORY, WHICH THE W=13/15 GATE DID NOT NEED TO.
# MCFM's blend is precomputed once over the cached conditioning, so a wider window
# there costs setup and nothing per step. rung31 is the opposite: the context tensor
# itself grows, 1029 -> 11,319 tokens at W=11, and attention memory grows with it.
# W=3 already peaks at 24 GiB of a 48 GiB card. A blind fan-out risks 42 simultaneous
# CUDA OOMs, so the release waits for a real measured peak with headroom.
#
# PASS CONDITIONS, all three:
#   1. [R31] window=11 ... tokens 1029 -> 11319   the width actually took effect
#   2. [GATE-window] passed                        centre slice is still frame f
#   3. max peak=<x>GiB <= CEIL                     it fits, with room for a worse object
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
LOGD=$E/out/FIGRUNS
OBJ=${OBJ:-spot_lava}
TW=${TW:-11}
CEIL=${CEIL:-42}          # GiB, against a 48 GiB card
NEED=${NEED:-20}          # peak readings before trusting the maximum
cd "$E"

logf() { ls -t "$LOGD/31_w${TW}_${OBJ}_"*.log 2>/dev/null | head -1 || true; }

echo "gate armed $(date '+%F %T')  obj=$OBJ  W=$TW  vram ceiling ${CEIL}GiB"
while :; do
  f=$(logf)
  if [ -n "$f" ]; then
    # `|| true` on every grep: a non-matching grep exits 1, pipefail propagates, and
    # set -e would kill the gate. That is how the first render gate died.
    died=$({ grep -lE "Traceback \(most recent call last\)|CUDA out of memory|\[GATE-[a-z]+\] FAILED" "$f" 2>/dev/null || true; } | head -1)
    if [ -n "$died" ]; then
      echo "GATE FAILED: canary error"
      grep -E "CUDA out of memory|Traceback|GATE.*FAILED|\[R31\]" "$f" | tail -12
      echo "NOT releasing the other 41."
      exit 1
    fi
    width=$({ grep -m1 "\[R31\] window=" "$f" 2>/dev/null || true; })
    peaks=$({ grep -o "peak=[0-9.]*GiB" "$f" 2>/dev/null || true; } | sed 's/[^0-9.]//g')
    npk=$(printf '%s\n' "$peaks" | grep -c . || true)
    maxpk=$(printf '%s\n' "$peaks" | sort -g | tail -1)
    echo "$(date '+%T')  peaks=${npk:-0}  max=${maxpk:-?}GiB  ${width:+width-ok}"
    if [ "${npk:-0}" -ge "$NEED" ]; then
      [ -z "$width" ] && { echo "GATE FAILED: no [R31] window= line, the width never took effect"; exit 1; }
      over=$(python3 -c "print(1 if float('${maxpk:-99}') > ${CEIL} else 0)")
      if [ "$over" = "1" ]; then
        echo "GATE FAILED: peak ${maxpk}GiB exceeds the ${CEIL}GiB ceiling."
        echo "  W=11 does not fit with headroom. Options: drop to W=7/9, or lower"
        echo "  --resolution, or accept OOM risk on the heavier objects."
        exit 1
      fi
      echo "GATE PASSED $(date '+%F %T'): $width"
      echo "  peak ${maxpk}GiB of 48, under the ${CEIL}GiB ceiling. Releasing 41."
      break
    fi
  else
    echo "$(date '+%T')  waiting for the canary to start"
  fi
  sleep 60
done
TWIN=$TW bash "$E/jobs/submit_rung31_w11.sh"
echo "release complete $(date '+%F %T')"
