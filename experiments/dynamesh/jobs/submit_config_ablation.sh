#!/bin/bash
# submit_config_ablation.sh -- the CONFIGURATION ablation (3DV supplement), all 24
# objects of batches A+B+C, rung27 backbone held fixed.
#
# TWO AXES, ONE BACKBONE. rung27 (qkvo+sa, rank 4, seed 42, 30 epochs, l1+lpips)
# is the only rung used. Sweeping 28/29/30/31/33 as well would confound the config
# axis with the KL / dual-attention axis and the table would answer neither question.
#
#   window size   W=1 mcfm=None (have 24/24)  W=3 v2_D (have 24/24)  W=5 v2_E (0/24)
#   flavour       temporal->spatial v2_D = OURS   spatial->temporal st_D   joint v3_D
#
# WHY v2_D IS "temporal then spatial". blend_conds() is applied to the cached DINOv3
# conditioning at rung27_selfattn_lora.py:1505, BEFORE the flow runs; the model's own
# spatial cross-attention then consumes the blended tokens. The ordering is a property
# of the pipeline, not of the blend. mcfm_blend's ts_* variant stacks a SECOND explicit
# spatial stage inside the blend and is therefore a different operator, not this cell.
#
# THE OBJECT SET IS THE 24 IN out/texel_r19_params.tsv, i.e. the same set the main
# results table averages over. MESH / GT-RENDER-DIR / NFR are read from each object's
# own pinned rung19 config.json -- never globbed, never assumed. pumpkin_rot alone is
# 121 frames on the _guan mesh and targets; hardcoding hero paths would train it
# against the wrong targets, complete successfully, and be wrong.
#
# CONSTRAINT a40|L40S, NOT the template's a40|L40S|a30. Every crashed st_D/v3_D run
# on disk died with `GLIBC_2.32 not found` importing o_voxel._C -- the trellis2 env is
# built for a40/L40S only.
# MEM 32G, NOT the template's 96G. Same script, same flags as fig27.sbatch, whose
# measured peak RSS is 15.3G; 96G costs 24 of 34 billing units under the group TRES
# cap and would throttle this fleet to ~30 concurrent for no reason.
#
# Idempotent: a cell with a finished checkpoint, or a job already queued for it, is
# skipped. Re-running this after failures resubmits only what is still missing.
# DRY=1 prints the work list and submits nothing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
MODES="${MODES:-v2_E st_D v3_D}"   # env-overridable: the window ablation passes v2_F v2_G
cd "$E"

note() {   # jid name export script desc
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
}

# ---- preflight: emit obj<TAB>nfr<TAB>mesh<TAB>gtdir<TAB>mode for every MISSING cell
squeue -u "$USER" -h -o "%j" > /tmp/_ca_queue.$$ || true
WORK=$(MODES="$MODES" python3 - <<'PY'
import json, os, glob
roster = [l.split('\t') for l in
          open('out/texel_r19_params.tsv').read().splitlines()[1:] if l.strip()]
names = [o for o, _, _ in roster]
modes = os.environ['MODES'].split()
queued = set(open(glob.glob('/tmp/_ca_queue.*')[0]).read().split())

# what already has a usable checkpoint, keyed on the EXACT /data/<obj>/ in gt_dir.
# substring matching here is a trap: 'dragon_mush' is a prefix of 'dragon_mush2'
# and silently steals its runs.
have = set()
for d in glob.glob('runs/rung27_*'):
    c = os.path.join(d, 'config.json')
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    # COMPLETION GATE: final_eval.json, not ckpts/lora_best.pt. The trainer rewrites
    # lora_best.pt after every improving epoch, so a checkpoint exists from epoch 1 --
    # treating that as "already have this cell" would make a re-run silently skip a cell
    # whose training died halfway. final_eval.json is written once, at the end.
    if not os.path.exists(os.path.join(d, 'final_eval.json')):
        continue
    if cfg.get('context_window'):
        continue
    gt = cfg.get('gt_dir') or ''
    obj = next((o for o in names if f'/data/{o}/' in gt), None)
    if obj:
        have.add((obj, str(cfg.get('mcfm'))))

for obj, run, nfr in roster:
    cfg = json.load(open(os.path.join(run, 'config.json')))
    for m in modes:
        if (obj, m) in have or f'm3_{obj}_{m}' in queued:
            continue
        print('\t'.join([obj, nfr, cfg['mesh'], cfg['gt_render_dir'], m]))
PY
)
rm -f /tmp/_ca_queue.$$

N=$(printf '%s\n' "$WORK" | grep -c . || true)
echo "== config ablation: ${N} cells missing (of $((24 * $(echo $MODES | wc -w))))"
[ "$N" = "0" ] && { echo "nothing to do"; exit 0; }
printf '%s\n' "$WORK" | awk -F'\t' '{printf "   %-26s %-6s %s\n",$1,$5,$2"fr"}'
[ -n "${DRY:-}" ] && { echo "DRY=1 — nothing submitted"; exit 0; }

echo; echo "== submitting =="
while IFS=$'\t' read -r OBJ NFR MESH GTDIR MODE; do
  [ -z "$OBJ" ] && continue
  EXP="OBJ=${OBJ},NFR=${NFR},MODE=${MODE},MESH=${MESH},GTDIR=${GTDIR}"
  J=$(sbatch --parsable --job-name="m3_${OBJ}_${MODE}" \
       --constraint="a40|L40S" --mem=32G \
       --export=ALL,"$EXP" "$E/jobs/rung27_mcfm_mode.sbatch")
  note "$J" "m3_${OBJ}_${MODE}" "$EXP" "jobs/rung27_mcfm_mode.sbatch" \
    "CONFIG ABLATION (3DV supp): rung27 qkvo+sa r4 s42 30ep + MCFM ${MODE} on ${OBJ}, ${NFR}fr; backbone identical to the ${OBJ} mcfm=None and v2_D runs in every config field except mcfm, so the cells are comparable; a40|L40S only (a30 dies on GLIBC_2.32 in o_voxel._C); out=$E/out/r27m3_${J}.log; PASS: log has '=== Done' and runs/rung27_l1_lp_mcfm${MODE}_all_qkvo+sa_r4_s42_*/ckpts/lora_best.pt exists for ${OBJ}"
  echo "  $J  m3_${OBJ}_${MODE}"
done <<< "$WORK"

echo; echo "next: jobs/submit_config_ablation_texel.sh once these finish (idempotent, skips unfinished cells)"
