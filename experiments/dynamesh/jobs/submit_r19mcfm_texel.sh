#!/bin/bash
# submit_r19mcfm_texel.sh -- texel metrics for the reversed ladder's "+temporal+CA"
# row (rung 19 trained WITH --mcfm). Run repeatedly: measures whatever has finished
# training, skips the rest. Safe while the r26ca fleet is still going.
#
# TAG is <obj>_r19mcfm. build_results_table.py and build_component_ladder.py both read
# an explicit arm list and never glob out/TEXEL, so this tag cannot contaminate
# Table 1 or the main results table.
#
# ROW 2 FALLS OUT OF THIS. render_rung27_orbit.py decodes the frozen and adapted arms
# from the same blended conditioning, so each json also carries frozen_* -- that IS
# frozen+MCFM, the reversed ladder's second row, at no extra GPU cost.
# DRY=1 prints the work list and submits nothing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
MODE="${MODE:-v2_D}"; RUNG=19
cd "$E"; mkdir -p out/TEXEL

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

squeue -u "$USER" -h -o "%j" > /tmp/_r19t_q.$$ || true
WORK=$(MODE="$MODE" python3 - <<'PY'
import json, os, glob
roster=[l.split('\t') for l in open('out/texel_r19_params.tsv').read().splitlines()[1:] if l.strip()]
names=[o for o,_,_ in roster]; mode=os.environ['MODE']
queued=set(open(glob.glob('/tmp/_r19t_q.*')[0]).read().split())
# a cell is measurable once its training run has a checkpoint AND final_eval.json
ready={}
for d in glob.glob('runs/rung19_*'):
    c=os.path.join(d,'config.json')
    if not os.path.exists(c): continue
    try: cfg=json.load(open(c))
    except Exception: continue
    if str(cfg.get('mcfm'))!=mode: continue
    if not os.path.exists(os.path.join(d,'final_eval.json')): continue
    if not os.path.exists(os.path.join(d,'ckpts','lora_best.pt')): continue
    gt=cfg.get('gt_dir') or ''
    o=next((o for o in names if f'/data/{o}/' in gt), None)
    if o: ready[o]=d
for obj,run,nfr in roster:
    tag=f'{obj}_r19mcfm'
    if obj not in ready: continue
    p=f'out/TEXEL/{tag}.json'
    if os.path.exists(p) and os.path.getsize(p)>0: continue
    if f'tx19_{obj}' in queued: continue
    print('\t'.join([obj,nfr,tag]))
PY
)
rm -f /tmp/_r19t_q.$$
N=$(printf '%s\n' "$WORK" | grep -c . || true)
echo "== r19+mcfm(${MODE}) texel: ${N} cells measurable now"
[ "$N" = "0" ] && { echo "nothing measurable yet"; exit 0; }
printf '%s\n' "$WORK" | awk -F'\t' '{printf "   %-26s -> %s\n",$1,$3}'
[ -n "${DRY:-}" ] && { echo "DRY=1 -- nothing submitted"; exit 0; }

echo; echo "== submitting =="
while IFS=$'\t' read -r OBJ NFR TAG; do
  [ -z "$OBJ" ] && continue
  EXP="OBJ=${OBJ},MODE=${MODE},TAG=${TAG},NFR=${NFR},RUNG=${RUNG}"
  J=$(sbatch --parsable --job-name="tx19_${OBJ}" --export=ALL,"$EXP" \
       "$E/jobs/texel_after_train.sbatch")
  note "$J" "tx19_${OBJ}" "$EXP" "jobs/texel_after_train.sbatch" \
    "REVERSED LADDER texel: rung19 + MCFM ${MODE} on ${OBJ}, ${NFR}fr -> out/TEXEL/${TAG}.json; its frozen_* fields also give frozen+MCFM (ladder row 2); PASS: log has '=== Done' and out/TEXEL/${TAG}.json non-empty"
  echo "  $J  tx19_${OBJ}"
done <<< "$WORK"
