#!/bin/bash
# submit_fullrate.sh -- one CPU job per object for the full-rate (150-frame) table.
# Idempotent: an object with a written json, or a job already queued for it, is skipped.
# DRY=1 prints the work list and submits nothing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"; mkdir -p out/FULLRATE

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

squeue -u "$USER" -h -o "%j" > /tmp/_fr_q.$$ || true
OBJS=$(python3 - <<'PY'
import json,os,glob
B='/net/projects/ranalab/rajhansini/baselines4d'
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
objs=sorted({k.split('|')[0] for k in json.load(open(f'{B}/results/video_metrics_all.json'))['per_cell']})
queued=set(open(glob.glob('/tmp/_fr_q.*')[0]).read().split())
for o in objs:
    p=f'{E}/out/FULLRATE/{o}.json'
    if os.path.exists(p) and os.path.getsize(p)>0: continue
    if f'fr_{o}' in queued: continue
    print(o)
PY
)
rm -f /tmp/_fr_q.$$
N=$(printf '%s\n' "$OBJS" | grep -c . || true)
echo "== full-rate: ${N}/24 objects to run"
[ "$N" = "0" ] && { echo "nothing to do"; exit 0; }
[ -n "${DRY:-}" ] && { printf '%s\n' "$OBJS"; echo "DRY=1 -- nothing submitted"; exit 0; }

for OBJ in $OBJS; do
  [ -z "$OBJ" ] && continue
  J=$(sbatch --parsable --job-name="fr_${OBJ}" --export=ALL,OBJ="${OBJ}" \
       "$E/jobs/fullrate_obj.sbatch")
  note "$J" "fr_${OBJ}" "OBJ=${OBJ}" "jobs/fullrate_obj.sbatch" \
    "FULL-RATE video metrics (Table B, 3DV): all 150 frames x 4 cameras x {GT,frozen,ours,MeshNCA,L4GM} on ${OBJ}; CPU-only, no GPU (NFS-read bound, ~2550 reads); out=$E/out/FULLRATE/${OBJ}.json; PASS: log has '=== Done' and out/FULLRATE/${OBJ}.json is non-empty"
  echo "  $J  fr_${OBJ}"
done
echo
echo "next: jobs/fullrate_table.py once these finish (validates the 21-instant reproduction first)"
