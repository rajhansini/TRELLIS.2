#!/bin/bash
# submit_window_hi.sh -- extend the WINDOW ablation to W = 13 (v2_H) and W = 15 (v2_I)
# across ALL 42 objects, batches A through E. 84 cells.
#
# WHAT HAD TO CHANGE FIRST. mcfm_blend.py's _OFFSETS stopped at 'G' (W=11), so these
# two windows were not merely unrun, they were unrunnable: parse_mode() would have
# raised KeyError on 'H'. H and I are added there as symmetric offset tuples, and
# verified on CPU before any job was submitted -- parse_mode returns the right width
# and centre for both, blend_conds preserves shape and finiteness on a 150-frame and
# a 121-frame sequence, and on an 8-frame sequence, which is the real edge case: a
# window WIDER than the sequence, where every neighbour clamps.
#
# WHY ALL 42 AND NOT THE 24. W=3/5/7/11 are all 42/42 across A-E, so anything less
# here would make the last two points of the curve average over a different object
# set than the first four -- which is the same defect submit_texel_r19.sh exists to
# avoid on the ablation table.
#
# EXPECT TIMEOUTS. fig27.sbatch has a 4h wall because partition MaxTime IS 4h, and
# 30 epochs takes 3-6h. TIMEOUT is the NORMAL outcome, not a failure: the run resumes
# from ckpts on requeue, and watchdog.py already covers the '27_' job-name prefix, so
# these are picked up without a new KIND entry. A cell is "done" only when its run
# directory holds final_eval.json.
#
#   MODES=v2_H,v2_I   restrict the windows (default: both)
#   ONLY=obj1,obj2    restrict the objects
#   LIMIT=n           submit at most n jobs (canary runs)
#   DRY=1             print the work list, submit nothing
#   NICE=n            deprioritise against other work
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
cd "$E"
[ -s "$TSV" ] || { echo "missing $TSV"; exit 1; }
MODES=${MODES:-v2_H,v2_I}
mkdir -p "$E/out/FIGRUNS"

# The operator must be able to express the window before 84 jobs go looking for it.
for M in ${MODES//,/ }; do
  python3 -c "import sys; sys.path.insert(0,'$E'); import mcfm_blend as m; m.parse_mode('$M')" \
    || { echo "FATAL: mcfm_blend.parse_mode('$M') failed -- fix _OFFSETS before submitting"; exit 1; }
done

note() {
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
}

# A run dir WITHOUT final_eval.json is a partially trained cell, not a finished one:
# fig27.sbatch resumes from ckpts, so those are resubmitted rather than skipped.
finished() {
  python3 - "$E" "$1" "$2" <<'PY'
import json,sys,glob,os
E,obj,mode=sys.argv[1:4]
for c in glob.glob(f'{E}/runs/rung27_l1_lp_mcfm{mode}_all_qkvo+sa_r4_s42_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('epochs')!=30 or str(d.get('mcfm'))!=mode: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if os.path.exists(os.path.join(os.path.dirname(c),'final_eval.json')):
        print('yes'); break
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_whi.$$ || true
N=0; S=0; SKIP=0
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  [ -z "$OBJ" ] && continue
  [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
  for MODE in ${MODES//,/ }; do
    N=$((N+1))
    [ -n "$(finished "$OBJ" "$MODE")" ] && { SKIP=$((SKIP+1)); continue; }
    JN="27_${MODE}_${OBJ}"
    grep -qx "$JN" /tmp/_whi.$$ && { SKIP=$((SKIP+1)); continue; }
    # Same four preconditions submit_window_de.sh checks. GTDIR (col 5) is the raw
    # video; GTREN (col 6) is the back-projected target, and it is GTREN that
    # fig27.sbatch needs -- passing col 5 sent all 53 of that batch into
    # "GATE-align FAILED: render_mask.npy is missing" after ~53s.
    [ -f "$MESH" ]  || { echo "   MISSING MESH        $OBJ -> $MESH"; continue; }
    [ -d "$GTDIR" ] || { echo "   MISSING GTDIR       $OBJ -> $GTDIR"; continue; }
    [ -d "$GTREN" ] || { echo "   MISSING GTREN       $OBJ -> $GTREN"; continue; }
    [ -f "$(dirname "$GTREN")/render_mask.npy" ] || { echo "   MISSING render_mask $OBJ"; continue; }
    if [ -n "${DRY:-}" ]; then
      printf '   %-1s %-24s %-5s nfr=%s\n' "$B" "$OBJ" "$MODE" "$NFR"; continue
    fi
    [ -n "${LIMIT:-}" ] && [ "$S" -ge "$LIMIT" ] && continue
    W=$(python3 -c "import sys; sys.path.insert(0,'$E'); import mcfm_blend as m; print(len(m.parse_mode('$MODE')[1]))")
    EXP="OBJ=${OBJ},MODE=${MODE},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN}"
    J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" \
         --export=ALL,"$EXP" jobs/fig27.sbatch)
    note "$J" "$JN" "$EXP" "jobs/fig27.sbatch" \
      "WINDOW ablation W=${W}: rung27 qkvo+sa r4 + MCFM ${MODE} (temporal-only, ${W}-frame window), ${OBJ} batch ${B}, ${NFR}fr, seed42; log out/FIGRUNS/${JN}_${J}.log; TIMEOUT at the 4h wall is expected and resumes from ckpts; PASS: log has '[FINAL] rung' and the run's final_eval.json exists"
    echo "  $J  $JN"
    S=$((S+1))
  done
done < "$TSV"
rm -f /tmp/_whi.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
exit 0
