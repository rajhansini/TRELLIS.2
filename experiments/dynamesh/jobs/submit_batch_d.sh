#!/bin/bash
# BATCH D — the 7-arm supplementary ladder on all 11 batch-D assets. 77 jobs.
#   qkvo    = rung19 (cross-attention only)      qkvo+sa = rung27 (adds self-attention)
#   MODE empty = no blend; otherwise the MCFM variant.
# Idempotent: an (obj,arm) whose run dir already has final_eval.json, or that is already
# queued, is skipped. final_eval.json and NOT ckpts/lora_best.pt -- the trainer rewrites
# lora_best.pt every improving epoch, so a checkpoint exists from epoch 1 and using it as
# the completion test would silently skip a cell whose training died halfway.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="moai_silver moai_animated gargoyle_spiral gargoyle_effect_one airplane_blub airplane_red_cracks teddy_bleach teddy_fusion goat_burnt goat_flower goat_clay"
# armkey : TARGETS : MODE
ARMS="r19:qkvo: r19v2D:qkvo:v2_D r27:qkvo+sa: r27v2D:qkvo+sa:v2_D r27v2E:qkvo+sa:v2_E r27stD:qkvo+sa:st_D r27v3D:qkvo+sa:v3_D"
squeue -u "$USER" -h -o "%j" > /tmp/_bd_q.$$ || true
N=0; SK=0
for OBJ in $OBJS; do
  MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
  GTDIR="$E/out/gt_targets_$OBJ/frames"
  [ -f "$MESH" ] || { echo "ABORT: missing $MESH"; exit 1; }
  [ -d "$GTDIR" ] || { echo "ABORT: missing $GTDIR"; exit 1; }
  for A in $ARMS; do
    KEY=${A%%:*}; REST=${A#*:}; TARGETS=${REST%%:*}; MODE=${REST#*:}
    NAME="bd_${OBJ}_${KEY}"
    grep -qx "$NAME" /tmp/_bd_q.$$ && { SK=$((SK+1)); continue; }
    if [ -n "${MODE}" ]; then RUNPAT="runs/rung*_l1_lp_mcfm${MODE}_all_${TARGETS}_r4_s42_*"; \
                        else RUNPAT="runs/rung*_l1_lp_all_${TARGETS}_r4_s42_*"; fi
    DONE=0
    for d in $RUNPAT; do
      [ -f "$d/final_eval.json" ] || continue
      grep -q "/data/$OBJ/" "$d/config.json" 2>/dev/null && DONE=1 && break
    done
    [ "$DONE" = "1" ] && { SK=$((SK+1)); continue; }
    EXP="OBJ=${OBJ},NFR=150,TARGETS=${TARGETS},MODE=${MODE},MESH=${MESH},GTDIR=${GTDIR}"
    [ -n "${DRY:-}" ] && { echo "DRY $NAME  TARGETS=$TARGETS MODE=${MODE:-none}"; N=$((N+1)); continue; }
    J=$(sbatch --parsable --job-name="$NAME" --export=ALL,"$EXP" "$E/jobs/batch_d_arm.sbatch")
    python3 - "$MAN" "$J" "$NAME" "$EXP" "jobs/batch_d_arm.sbatch" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
    $JL add "$J" "$NAME" "BATCH D supp ladder: ${TARGETS} r4 s42 30ep mcfm=${MODE:-none} on ${OBJ}, 150fr, 2D-copy targets out/gt_targets_${OBJ}/frames | out/bd_${J}.log | PASS: log has '=== Done' and the run dir gains final_eval.json"
    echo "  $J  $NAME"
    N=$((N+1))
  done
done
rm -f /tmp/_bd_q.$$
echo "submitted $N, skipped $SK"
