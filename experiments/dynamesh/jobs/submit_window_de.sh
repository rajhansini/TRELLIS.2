#!/bin/bash
# submit_window_de.sh -- extend the WINDOW ablation from 24 objects to all 42 by
# training the wide windows on batches D and E.
#
# STATE THIS FILLS. All 18 batch D/E objects already have targets, W=1 (rung27, no blend)
# and W=3 (v2_D). What is missing is W=5 (v2_E), W=7 (v2_F) and W=11 (v2_G) -- 53 cells,
# since moai_silver already has W=5.
#
# Objects, frame counts, mesh and target paths come from jobs/rung37_objects.tsv, the
# same registry submit_rung37.sh reads. Nothing is derived from the object name.
#
# A run directory that exists WITHOUT final_eval.json is a partially trained cell, not a
# finished one: fig27.sbatch resumes from ckpts, so those are resubmitted rather than
# skipped. Four v2_E cells are in exactly that state.
#
# DRY=1 prints the work list. ONLY=<obj>[,<obj>] restricts. MODES=v2_F,v2_G restricts.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
cd "$E"
[ -s "$TSV" ] || { echo "missing $TSV"; exit 1; }
MODES=${MODES:-v2_E,v2_F,v2_G}

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

# is there a FINISHED run (final_eval.json) for this obj+mode?
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

squeue -u "$USER" -h -o "%j" > /tmp/_wd.$$ || true
N=0; S=0
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  case "$B" in D|E) ;; *) continue ;; esac
  [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
  for MODE in ${MODES//,/ }; do
    N=$((N+1))
    [ -n "$(finished "$OBJ" "$MODE")" ] && continue
    JN="27_${MODE}_${OBJ}"
    grep -qx "$JN" /tmp/_wd.$$ && { echo "   queued  $JN"; continue; }
    [ -f "$MESH" ] || { echo "   MISSING MESH $OBJ -> $MESH"; continue; }
    [ -d "$GTDIR" ] || { echo "   MISSING GTDIR $OBJ -> $GTDIR"; continue; }
    [ -d "$GTREN" ] || { echo "   MISSING GTREN $OBJ -> $GTREN"; continue; }
    [ -f "$(dirname "$GTREN")/render_mask.npy" ] || { echo "   MISSING render_mask $OBJ"; continue; }
    if [ -n "${DRY:-}" ]; then printf '   %-1s %-24s %-6s nfr=%s\n' "$B" "$OBJ" "$MODE" "$NFR"; continue; fi
    # fig27.sbatch feeds GTDIR to --gt-render-dir, which needs the BACK-PROJECTED
    # TARGETS (col 6, gt_render_dir), not the raw video frames (col 5, gt_dir).
    # Passing col 5 sent all 53 jobs into "GATE-align FAILED: render_mask.npy is
    # missing" after ~53s, because the mask sits next to the targets.
    EXP="OBJ=${OBJ},MODE=${MODE},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN}"
    J=$(sbatch --parsable --job-name="$JN" --export=ALL,"$EXP" jobs/fig27.sbatch)
    note "$J" "$JN" "$EXP" "jobs/fig27.sbatch" \
      "WINDOW ablation batch ${B}: rung27 qkvo+sa r4 + MCFM ${MODE}, ${NFR}fr, ${OBJ}; PASS: log has '[FINAL] rung' and the run's final_eval.json exists (log out/FIGRUNS/${JN}_${J}.log)"
    echo "  $J  $JN"
    S=$((S+1))
  done
done < "$TSV"
rm -f /tmp/_wd.$$
echo; echo "cells considered ${N}; submitted ${S}"
exit 0
