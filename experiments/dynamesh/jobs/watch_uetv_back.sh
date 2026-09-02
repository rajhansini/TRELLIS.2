#!/bin/bash
set -u
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
TAGS=(t1_y180 t1_y215 t2_y180 t2_y215)
declare -A JID DONE; i=1
for t in "${TAGS[@]}"; do JID[$t]=$(awk -v n=$i '{print $n}' $E/out/FIGRUNS/uetv_back_jobids.txt); DONE[$t]=0; i=$((i+1)); done
ok(){ [ "$(ls $E/out/uet_$1/frames/*.png 2>/dev/null | wc -l)" = 150 ]; }
while true; do
  all=1
  for t in "${TAGS[@]}"; do
    [ "${DONE[$t]}" = 1 ] && continue; all=0
    st=$(sacct -j "${JID[$t]}" -X --format=State%16 -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in RUNNING|PENDING|"") ;;
      COMPLETED) ok "$t" && echo "$(date '+%T') PASS $t" || echo "$(date '+%T') WARN $t no frames"; DONE[$t]=1 ;;
      FAILED) echo "$(date '+%T') FAILED $t (job ${JID[$t]})"; DONE[$t]=1 ;;
      *) ok "$t" && { echo "$(date '+%T') PASS $t after $st"; DONE[$t]=1; } ;;
    esac
  done
  [ "$all" = 1 ] && { echo "ALL TERMINAL"; break; }; sleep 60
done
