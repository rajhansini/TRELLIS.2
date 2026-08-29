#!/bin/bash
# submit_rung37.sh -- rung37 (per-position temporal LoRA) over batches A-E.
#
# rung37 = rung27's cross-attn + self-attn LoRA, PLUS a third temporal LoRA on a
# reused copy of the frozen cross-attention whose attention is restricted to the
# SAME SPATIAL POSITION across a 3-frame window. That is MCFM v2_D's geometry with
# trained Q/K instead of a fixed softmax, so it initialises exactly at MCFM and can
# only move away from it. Stamps rung:37 + temporal_geometry:per_position, so it can
# never share a directory or resume a checkpoint with rung31 (joint spatio-temporal).
#
# Object list, meshes and target dirs: jobs/rung37_objects.tsv (42 objects, verified
# present). pumpkin_rot is 121 frames; every other object is 150 -- read NFR from the
# file, never hardcode it.
#
#   BATCH=A,B,C   restrict to those batches (default: all)
#   ONLY=obj1,obj2  restrict to named objects
#   DRY=1         print the work list and submit nothing
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
cd "$E"
[ -s "$TSV" ] || { echo "missing $TSV"; exit 1; }

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

squeue -u "$USER" -h -o "%j" > /tmp/_r37q.$$ || true
N=0; SUB=0
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  [ -n "${BATCH:-}" ] && [[ ",$BATCH," != *",$B,"* ]] && continue
  [ -n "${ONLY:-}"  ] && [[ ",$ONLY,"  != *",$OBJ,"* ]] && continue
  N=$((N+1))
  # idempotent: skip a finished cell or one already queued
  if compgen -G "runs/rung37_l1_lp_tw3bpp_all_qkvo+sa_r4_s42_*/final_eval.json" >/dev/null 2>&1; then
    for d in runs/rung37_l1_lp_tw3bpp_all_qkvo+sa_r4_s42_*/; do
      [ -f "$d/final_eval.json" ] || continue
      # A SMOKE RUN ALSO WRITES final_eval.json. Without the epochs/n_frames check a
      # 2-epoch 8-frame probe counts as "done" and the real run is never submitted --
      # which is exactly what happened to chair_moss on the first pass.
      python3 -c "
import json,sys;c=json.load(open('$d/config.json'))
ok = ('/data/$OBJ/' in (c.get('gt_dir') or '')) and c.get('epochs')==30 and str(c.get('n_frames'))=='$NFR'
sys.exit(0 if ok else 1)" 2>/dev/null && { echo "  skip $OBJ (done)"; continue 2; }
    done
  fi
  grep -qx "r37_${OBJ}" /tmp/_r37q.$$ && { echo "  skip $OBJ (queued)"; continue; }
  if [ -n "${DRY:-}" ]; then printf '   %-2s %-24s %sfr  %s\n' "$B" "$OBJ" "$NFR" "$(basename "$MESH")"; continue; fi
  EXP="OBJ=${OBJ},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN}"
  # NICE deprioritises these so they never outrank the user's own higher-priority
  # work: a niced job only takes a slot nothing else wants. NOTE it is a submit-line
  # flag, so a REQUEUE loses it and the job returns at normal priority.
  J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="r37_${OBJ}" --export=ALL,"$EXP" "$E/jobs/rung37_perpos.sbatch")
  note "$J" "r37_${OBJ}" "$EXP" "jobs/rung37_perpos.sbatch" \
    "RUNG37 batch ${B}: per-position temporal LoRA (window 3, branch both, gate 0) + cross/self LoRA, r4 s42 30ep, ${OBJ} ${NFR}fr; out=$E/out/r37_${J}.log; PASS: log has '=== Done' and runs/rung37_l1_lp_tw3bpp_all_qkvo+sa_r4_s42_*/final_eval.json exists for ${OBJ}"
  echo "  $J  r37_${OBJ}  (batch $B)"
  SUB=$((SUB+1))
done < "$TSV"
rm -f /tmp/_r37q.$$
echo
echo "matched ${N} objects; submitted ${SUB}"
[ -n "${DRY:-}" ] && echo "DRY=1 -- nothing submitted"
echo "then: measure with jobs/texel_after_train.sbatch (RUNG=37, MODE=None, TAG=<obj>_r37)"
