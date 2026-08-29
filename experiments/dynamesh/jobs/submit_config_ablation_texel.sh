#!/bin/bash
# submit_config_ablation_texel.sh -- texel flicker/accel/drift for the configuration
# ablation cells. Run it repeatedly: it measures whatever has finished training and
# skips everything else. Safe to re-run while the fleet is still going.
#
# CELL -> out/TEXEL FILE.  Two of the five cells need no new measurement at all,
# because they are arms the main table already measured:
#     W=1  (mcfm=None)  -> <obj>_r27.json       24/24 already on disk
#     W=3  (v2_D, OURS) -> <obj>_r27mcfm.json   23/24 already on disk
#     W=5  (v2_E)       -> <obj>_w5.json        this script
#     spatial->temporal -> <obj>_stD.json       this script
#     joint spatio-temp -> <obj>_v3D.json       this script
#
# NEW ARM TAGS CANNOT CONTAMINATE THE MAIN TABLE. build_results_table.py builds
# out/TEXEL/<obj>_<arm>.json from an explicit arm list; it never globs the directory,
# so w5/stD/v3D are invisible to it. 'w5' follows the tag submit_window_ablation.sh
# already used for spot_lava.
#
# texel_after_train.sbatch resolves the run directory at RUNTIME from (RUNG, MODE, OBJ)
# by reading each candidate's config.json, and exits non-zero unless exactly one match
# has a checkpoint -- so no dependency edge is needed and a resumed/requeued training
# job cannot be measured under the wrong directory. It is also why the manifest export
# for these carries no RUN=.
#
# DRY=1 prints the work list and submits nothing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
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

squeue -u "$USER" -h -o "%j" > /tmp/_cat_queue.$$ || true
WORK=$(QF=/tmp/_cat_queue.$$ python3 - <<'PY'
import json, os, glob
roster = [l.split('\t') for l in
          open('out/texel_r19_params.tsv').read().splitlines()[1:] if l.strip()]
names = [o for o, _, _ in roster]
queued = set(open(os.environ['QF']).read().split())

# mode -> arm tag. v2_D is included so the one missing r27mcfm cell is picked up too.
ARM = {'v2_E': 'w5', 'st_D': 'stD', 'v3_D': 'v3D', 'v2_D': 'r27mcfm'}

# (obj, mcfm) -> run dir, for finished rung27 runs only. Exact /data/<obj>/ match:
# 'dragon_mush' is a prefix of 'dragon_mush2' and substring matching steals its runs.
have = {}
for d in glob.glob('runs/rung27_*'):
    c = os.path.join(d, 'config.json')
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if cfg.get('context_window'):
        continue
    # COMPLETION GATE: final_eval.json, NOT ckpts/lora_best.pt. The trainer rewrites
    # lora_best.pt after every epoch that improves, so a checkpoint exists from epoch 1
    # onward -- gating on it submitted 12 measurements against adapters that were still
    # training (epoch ~5 of 30) and would have written partially-trained numbers into the
    # ablation table. final_eval.json is written once, at the end of training.
    if not os.path.exists(os.path.join(d, 'final_eval.json')):
        continue
    if not os.path.exists(os.path.join(d, 'ckpts', 'lora_best.pt')):
        continue
    gt = cfg.get('gt_dir') or ''
    obj = next((o for o in names if f'/data/{o}/' in gt), None)
    if obj:
        have[(obj, str(cfg.get('mcfm')))] = d

for obj, run, nfr in roster:
    for mode, arm in ARM.items():
        tag = f'{obj}_{arm}'
        if os.path.exists(f'out/TEXEL/{tag}.json'):
            continue
        if (obj, mode) not in have:          # not trained yet -- a later run picks it up
            continue
        if f'texdep_{tag}' in queued:
            continue
        print('\t'.join([obj, nfr, mode, tag]))
PY
)
rm -f /tmp/_cat_queue.$$

N=$(printf '%s\n' "$WORK" | grep -c . || true)
echo "== measurable now: ${N} cells"
[ "$N" = "0" ] && { echo "nothing measurable yet — re-run when training finishes"; exit 0; }
printf '%s\n' "$WORK" | awk -F'\t' '{printf "   %-26s %-6s -> %s\n",$1,$3,$4}'
[ -n "${DRY:-}" ] && { echo "DRY=1 — nothing submitted"; exit 0; }

echo; echo "== submitting =="
while IFS=$'\t' read -r OBJ NFR MODE TAG; do
  [ -z "$OBJ" ] && continue
  EXP="OBJ=${OBJ},MODE=${MODE},TAG=${TAG},NFR=${NFR},RUNG=27"
  J=$(sbatch --parsable --job-name="texdep_${TAG}" --constraint="a40|L40S" \
       --export=ALL,"$EXP" "$E/jobs/texel_after_train.sbatch")
  note "$J" "texdep_${TAG}" "$EXP" "jobs/texel_after_train.sbatch" \
    "CONFIG ABLATION texel flicker/accel/drift for ${OBJ} rung27 mcfm=${MODE} (${NFR}fr); run dir resolved at runtime from config.json, not globbed; measured on decoder texels (base_color 0:3) with voxel coords asserted identical every frame, ODE only via --skip-render; out=$E/out/texdep_${J}.log; PASS: $E/out/TEXEL/${TAG}.json exists and is non-empty"
  echo "  $J  texdep_${TAG}"
done <<< "$WORK"
