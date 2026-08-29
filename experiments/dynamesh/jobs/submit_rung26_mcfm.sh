#!/bin/bash
# submit_rung26_mcfm.sh -- the "+ temporal + cross-attention" row of the REVERSED
# ablation ladder. One job per object. MODE defaults to v2_D (temporal-only, W=3),
# matching the mode of the existing rung27+MCFM row so the ladder is one operator
# throughout. Override with MODE=v2_E.
#
# Idempotent: a cell with a finished checkpoint, or a job already queued, is skipped.
# DRY=1 prints the work list and submits nothing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
MODE="${MODE:-v2_D}"
cd "$E"

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

squeue -u "$USER" -h -o "%j" > /tmp/_r26_q.$$ || true
WORK=$(MODE="$MODE" python3 - <<'PY'
import json, os, glob
roster=[l.split('\t') for l in open('out/texel_r19_params.tsv').read().splitlines()[1:] if l.strip()]
names=[o for o,_,_ in roster]
mode=os.environ['MODE']
queued=set(open(glob.glob('/tmp/_r26_q.*')[0]).read().split())

# finished cells, keyed on the EXACT /data/<obj>/ in gt_dir. Substring matching is a
# trap: 'dragon_mush' is a prefix of 'dragon_mush2' and would steal its runs.
have=set()
for d in glob.glob(f'runs/rung19_l1_lp_mcfm{mode}_all_qkvo_r4_s42_*'):
    if not os.path.exists(os.path.join(d,'final_eval.json')):   # written once, at the end
        continue
    try: cfg=json.load(open(os.path.join(d,'config.json')))
    except Exception: continue
    gt=cfg.get('gt_dir') or ''
    o=next((o for o in names if f'/data/{o}/' in gt), None)
    if o: have.add(o)

for obj,run,nfr in roster:
    if obj in have or f'r26ca_{obj}' in queued: continue
    cfg=json.load(open(os.path.join(run,'config.json')))
    print('\t'.join([obj,nfr,cfg['mesh'],cfg['gt_render_dir']]))
PY
)
rm -f /tmp/_r26_q.$$
N=$(printf '%s\n' "$WORK" | grep -c . || true)
echo "== reversed-ladder temporal+CA (mode ${MODE}): ${N}/24 cells missing"
[ "$N" = "0" ] && { echo "nothing to do"; exit 0; }
printf '%s\n' "$WORK" | awk -F'\t' '{printf "   %-26s %sfr\n",$1,$2}'
[ -n "${DRY:-}" ] && { echo "DRY=1 -- nothing submitted"; exit 0; }

echo; echo "== submitting =="
while IFS=$'\t' read -r OBJ NFR MESH GTDIR; do
  [ -z "$OBJ" ] && continue
  EXP="OBJ=${OBJ},NFR=${NFR},MODE=${MODE},MESH=${MESH},GTDIR=${GTDIR}"
  J=$(sbatch --parsable --job-name="r26ca_${OBJ}" --export=ALL,"$EXP" \
       "$E/jobs/rung26_mcfm_ca.sbatch")
  note "$J" "r26ca_${OBJ}" "$EXP" "jobs/rung26_mcfm_ca.sbatch" \
    "REVERSED LADDER row 3 (3DV): rung26 --targets qkvo (= rung 19, cross-attn LoRA only) + MCFM ${MODE} on ${OBJ}, ${NFR}fr, r4 s42 30ep l1+lpips; isolates the temporal component introduced BEFORE the adapters; a40|L40S only; out=$E/out/r26ca_${J}.log; PASS: log has '=== Done' and runs/rung19_l1_lp_mcfm${MODE}_all_qkvo_r4_s42_*/final_eval.json exists for ${OBJ}"
  echo "  $J  r26ca_${OBJ}"
done <<< "$WORK"
echo; echo "next: measure with jobs/texel_after_train.sbatch -- each json also carries frozen_*, which fills the ladder's row 2 for free"
