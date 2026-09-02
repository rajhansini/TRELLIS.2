#!/bin/bash
# Monitor + resume for the BATCH H Fig.7 renders.
# PASS is checked on disk (150 frames in the tagged dir), because a COMPLETED job that
# wrote nothing is exactly the failure mode that wasted a round earlier today.
# TIMEOUT/NODE_FAIL resumes; FAILED does not — that is a bug, not a wall.
set -u
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
R1=$E/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_8393f186
R2=$E/runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_c33457a0
TAGS=(t1_y0 t1_y35 t1_y325 t2_y0 t2_y35 t2_y325)
declare -A RUNOF=([t1_y0]=$R1 [t1_y35]=$R1 [t1_y325]=$R1 [t2_y0]=$R2 [t2_y35]=$R2 [t2_y325]=$R2)
declare -A JID DONE NRES
i=1
for t in "${TAGS[@]}"; do JID[$t]=$(awk -v n=$i '{print $n}' $E/out/FIGRUNS/uetv_jobids.txt); DONE[$t]=0; NRES[$t]=0; i=$((i+1)); done
ok () { [ "$(ls $E/out/uet_$1/frames/*.png 2>/dev/null | wc -l)" = 150 ]; }
while true; do
  all=1
  for t in "${TAGS[@]}"; do
    [ "${DONE[$t]}" = 1 ] && continue
    all=0
    st=$(sacct -j "${JID[$t]}" -X --format=State%16 -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      RUNNING|PENDING|REQUEUED|SUSPENDED|"") ;;
      COMPLETED) if ok "$t"; then echo "$(date '+%T') PASS $t"; else echo "$(date '+%T') WARN $t COMPLETED but frames missing"; fi; DONE[$t]=1 ;;
      TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_ME*|CANCELLED*)
        if ok "$t"; then echo "$(date '+%T') PASS $t after $st"; DONE[$t]=1
        elif [ "${NRES[$t]}" -lt 4 ]; then
          NRES[$t]=$(( NRES[$t] + 1 ))
          y=${t##*_y}
          n=$(sbatch --parsable --job-name=uetv_$t --export=ALL,RUN=${RUNOF[$t]},TAG=$t,YAW=$y $E/jobs/uet_view.sbatch)
          echo "$(date '+%T') RESUME $t after $st -> $n (${NRES[$t]}/4)"; JID[$t]=$n
        else echo "$(date '+%T') GIVE UP $t"; DONE[$t]=1; fi ;;
      FAILED) echo "$(date '+%T') FAILED $t (job ${JID[$t]}) — not resumed; log out/FIGRUNS/uetv_${t}_${JID[$t]}.log"; DONE[$t]=1 ;;
      *) echo "$(date '+%T') $t state '$st'" ;;
    esac
  done
  [ "$all" = 1 ] && { echo "$(date '+%T') ALL TERMINAL"; break; }
  sleep 90
done
