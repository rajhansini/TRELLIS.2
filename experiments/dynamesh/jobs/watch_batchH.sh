#!/bin/bash
# BATCH H monitor WITH RESUME.
#
# 30 epochs against the hard 4h cap means TIMEOUT is the expected path, not a rare one, and
# 'bd_' is commented out of watchdog.py's KIND table (disabled 2026-08-26 for the batch-D
# objects), so nothing else will resubmit these. Resume is scoped to these two jobs only,
# rather than re-enabling the global prefix and changing behaviour for 77 unrelated runs.
#
# Resuming is safe because the trainer already supports it: per-epoch ckpts/lora_e*.pt are
# written atomically, it scans them newest-first with a torn-file fallback, and GATE-resume
# refuses a checkpoint trained against a different mesh. So re-running the identical command
# continues rather than restarting.
#
# FAILED is NOT resumed. A timeout means "needs more wall"; a failure means a bug, and
# resubmitting a bug just burns GPU in a loop -- which is how the 518-vs-960 pair would have
# spun forever had this been naive.
set -u
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
OBJS=(unicorn_extended_texture unicorn_extended_texture_2)
MAX_RESUME=6
declare -A JID DONE NRES
i=0
for o in "${OBJS[@]}"; do
  JID[$o]=$(awk -v n=$((i+1)) '{print $n}' "$E/out/FIGRUNS/batchH_jobids.txt"); DONE[$o]=0; NRES[$o]=0; i=$((i+1))
done

passed () {  # PASS = a finished run dir for this object, on disk
  local o=$1 d r
  d=$(grep -l "\"$o\"" "$E"/runs/*/config.json 2>/dev/null | head -1) || return 1
  [ -n "$d" ] || return 1
  r=$(dirname "$d")
  [ -s "$r/final_eval.json" ] && [ -s "$r/ckpts/lora_best.pt" ]
}

submit () {
  local o=$1
  sbatch --parsable --job-name=bd_${o} \
    --export=ALL,OBJ=$o,NFR=150,TARGETS=qkvo+sa,MODE=v2_D,\
MESH=$T2/data/$o/mesh/${o}_render_frame.obj,\
GTDIR=$E/out/gt_targets_${o}/frames \
    "$E/jobs/batch_d_arm.sbatch"
}

while true; do
  alldone=1
  for o in "${OBJS[@]}"; do
    [ "${DONE[$o]}" = 1 ] && continue
    alldone=0
    st=$(sacct -j "${JID[$o]}" -X --format=State%16 -n 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED|"") ;;
      COMPLETED)
        if passed "$o"; then echo "$(date '+%F %T') PASS $o (job ${JID[$o]})"; DONE[$o]=1
        else echo "$(date '+%F %T') WARN $o COMPLETED but no final_eval.json + lora_best.pt"; DONE[$o]=1; fi ;;
      TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_ME*|CANCELLED*)
        if passed "$o"; then echo "$(date '+%F %T') PASS $o after $st"; DONE[$o]=1
        elif [ "${NRES[$o]}" -lt "$MAX_RESUME" ]; then
          NRES[$o]=$(( NRES[$o] + 1 ))
          new=$(submit "$o")
          echo "$(date '+%F %T') RESUME $o after $st -> job $new (resume ${NRES[$o]}/$MAX_RESUME)"
          /net/projects/ranalab/rajhansini/joblog.sh add "$new" "bd_${o}" "BATCH H auto-resume ${NRES[$o]} after $st"
          JID[$o]=$new
        else
          echo "$(date '+%F %T') GIVE UP $o after $MAX_RESUME resumes"; DONE[$o]=1
        fi ;;
      FAILED)
        echo "$(date '+%F %T') FAILED $o (job ${JID[$o]}) — NOT resumed, this is a bug not a wall"
        echo "    log: $E/out/bd_${JID[$o]}.log"; DONE[$o]=1 ;;
      *) echo "$(date '+%F %T') $o unexpected state '$st'" ;;
    esac
  done
  [ "$alldone" = 1 ] && { echo "$(date '+%F %T') ALL TERMINAL"; break; }
  sleep 120
done
