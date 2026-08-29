#!/bin/bash
# submit_window_de_texel.sh -- texel temporal metrics for the WIDE windows on batches D+E.
#
# STATE THIS FILLS. After submit_window_de.sh, all 42 objects have PSNR/SSIM at
# W = 1/3/5/7/11. out/TEXEL is complete for A/B/C at all five widths and for D/E at
# W=1 and W=3 only, so flicker / acceleration / drift stop at 24 objects while
# reconstruction reaches 42. This fills the 54 missing cells (18 objects x W=5/7/11).
#
# RUN DIRS ARE RESOLVED FROM config.json, never globbed by name: several objects have
# more than one run per mode, and a name glob would silently pick the wrong one.
# Idempotent: a cell whose out/TEXEL/<tag>.json exists, or that is already queued, is skipped.
#
# DRY=1 prints the work list. ONLY=<obj>[,<obj>] restricts.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
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

resolve() {  # obj mode -> unique 30-epoch run dir with final_eval.json
  python3 - "$E" "$1" "$2" <<'PY'
import json,sys,glob,os
E,obj,mode=sys.argv[1:4]
hit=[]
for c in glob.glob(f'{E}/runs/rung27_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('rung')!=27 or d.get('epochs')!=30: continue
    if (d.get('mcfm') or 'none')!=mode: continue
    if f'/data/{obj}/' not in (d.get('mesh') or ''): continue
    r=os.path.dirname(c)
    if not os.path.exists(os.path.join(r,'final_eval.json')): continue
    hit.append(r)
print(hit[0] if len(hit)==1 else '')
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_wdt.$$ || true
N=0; S=0; MISS=""
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  case "$B" in D|E) ;; *) continue ;; esac
  [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
  for SPEC in "w5:v2_E" "w7:v2_F" "w11:v2_G"; do
    TAG=${OBJ}_${SPEC%%:*}; MODE=${SPEC##*:}
    N=$((N+1))
    [ -s "out/TEXEL/${TAG}.json" ] && continue
    grep -qx "tex_${TAG}" /tmp/_wdt.$$ && continue
    R=$(resolve "$OBJ" "$MODE")
    [ -z "$R" ] && { MISS="$MISS ${OBJ}/${MODE}"; continue; }
    if [ -n "${DRY:-}" ]; then printf '   %-24s %-4s %s\n' "$OBJ" "${SPEC%%:*}" "$(basename $R)"; continue; fi
    EXP="RUN=${R},TAG=${TAG},NFR=${NFR},SCRIPT=render_rung27_orbit.py"
    J=$(sbatch --parsable --job-name="tex_${TAG}" --export=ALL,"$EXP" jobs/texel_any.sbatch)
    note "$J" "tex_${TAG}" "$EXP" "jobs/texel_any.sbatch" \
      "WINDOW D/E texel: ${OBJ} ${MODE} (${SPEC%%:*}), ${NFR}fr -> out/TEXEL/${TAG}.json; PASS: json exists"
    echo "  $J  tex_${TAG}"
    S=$((S+1))
  done
done < "$TSV"
rm -f /tmp/_wdt.$$
echo; echo "cells considered ${N}; submitted ${S}"
[ -n "$MISS" ] && echo "UNRESOLVED:$MISS"
exit 0
