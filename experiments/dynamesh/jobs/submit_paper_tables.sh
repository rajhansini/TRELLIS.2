#!/bin/bash
# The ONLY arms the two paper tables need, across batches D and E.
#   rung19            -> ablation row 2  (and the frozen row, from its frozen_* fields)
#   rung27            -> ablation row 3
#   rung27 + MCFM v2_D-> ablation row 4  AND the comparison table's "DynaMesh (ours)" row
# The other four arms (r19+mcfm, v2_E, st_D, v3_D) feed the CONFIGURATION and
# temporal-first supplementary tables, not these two, and are deliberately not here.
# Idempotent: a cell whose run dir already has final_eval.json, or that is queued, is skipped.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2; E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json; JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
D="moai_silver moai_animated gargoyle_spiral gargoyle_effect_one airplane_blub airplane_red_cracks teddy_bleach teddy_fusion goat_burnt goat_flower goat_clay"
Eb="ivysaur_petal ivysaur_petal_2 blub_raurshaw blub_drying pegasus_shine pegaso_effect_1 mosaic_painting"
ARMS="r19:qkvo: r27:qkvo+sa: r27v2D:qkvo+sa:v2_D"
squeue -u "$USER" -h -o "%j" > /tmp/_pt_q.$$ || true
N=0; SK=0
for OBJ in $D $Eb; do
  MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
  GTDIR="$E/out/gt_targets_$OBJ/frames"
  [ -f "$MESH" ] || { echo "ABORT: missing $MESH"; exit 1; }
  [ -d "$GTDIR" ] || { echo "ABORT: missing $GTDIR"; exit 1; }
  for A in $ARMS; do
    KEY=${A%%:*}; R=${A#*:}; TARGETS=${R%%:*}; MODE=${R#*:}
    NAME="bd_${OBJ}_${KEY}"
    grep -qx "$NAME" /tmp/_pt_q.$$ && { SK=$((SK+1)); continue; }
    if [ -n "$MODE" ]; then P="runs/rung*_l1_lp_mcfm${MODE}_all_${TARGETS}_r4_s42_*"; \
                       else P="runs/rung*_l1_lp_all_${TARGETS}_r4_s42_*"; fi
    DONE=0
    for d in $P; do
      [ -f "$d/final_eval.json" ] || continue
      grep -q "/data/$OBJ/" "$d/config.json" 2>/dev/null && DONE=1 && break
    done
    [ "$DONE" = "1" ] && { SK=$((SK+1)); continue; }
    EXP="OBJ=${OBJ},NFR=150,TARGETS=${TARGETS},MODE=${MODE},MESH=${MESH},GTDIR=${GTDIR}"
    [ -n "${DRY:-}" ] && { echo "DRY $NAME"; N=$((N+1)); continue; }
    J=$(sbatch --parsable --job-name="$NAME" --export=ALL,"$EXP" "$E/jobs/batch_d_arm.sbatch")
    python3 - "$MAN" "$J" "$NAME" "$EXP" "jobs/batch_d_arm.sbatch" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
    $JL add "$J" "$NAME" "PAPER TABLES: ${TARGETS} r4 s42 30ep mcfm=${MODE:-none} on ${OBJ}, 150fr | out/bd_${J}.log | PASS: log has '=== Done' and the run dir gains final_eval.json"
    N=$((N+1))
  done
done
rm -f /tmp/_pt_q.$$
echo "submitted $N, skipped $SK (already complete)"
