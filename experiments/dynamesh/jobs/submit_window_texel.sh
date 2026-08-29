#!/bin/bash
# submit_window_texel.sh -- texel temporal metrics for the WIDE window widths.
#
# The window ablation already has PSNR/SSIM at W = 3/5/7/11 (24 objects each), and
# out/TEXEL carries flicker/acceleration/drift for W=3 (v2_D) and W=5 (v2_E) only. This
# fills W=7 (v2_F) and W=11 (v2_G) so the four-width table can carry temporal columns
# instead of reconstruction alone.
#
# RUN DIRS ARE RESOLVED FROM config.json, never globbed by name. Several objects have
# more than one run per mode -- spot_lava alone has three rung27 no-mcfm runs spanning
# 12 dB -- so a glob would silently drop a cell 4 dB from its neighbours into the table.
# Exact '/data/<obj>/' segment match, epochs==30, and final_eval.json present; anything
# ambiguous is reported as UNRESOLVED rather than guessed.
#
# Object list and per-object frame count come from out/texel_r19_params.tsv, the same
# 24 the paper's tables are restricted to.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
TSV=out/texel_r19_params.tsv
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

resolve() {  # obj mode -> run dir, or empty when absent or ambiguous
  python3 - "$E" "$1" "$2" <<'PY'
import json,sys,glob,os
E,obj,mode=sys.argv[1:4]
hit=[]
for c in glob.glob(f'{E}/runs/rung27_l1_lp_mcfm{mode}_all_qkvo+sa_r4_s42_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('rung')!=27 or d.get('epochs')!=30: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    r=os.path.dirname(c)
    if not os.path.exists(os.path.join(r,'final_eval.json')): continue
    hit.append(r)
print(hit[0] if len(hit)==1 else '')
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_wt.$$ || true
N=0; S=0; MISS=""
while IFS=$'\t' read -r OBJ RUN NFR; do
  [ "$OBJ" = "object" ] && continue
  for SPEC in "w7:v2_F" "w11:v2_G"; do
    TAG=${OBJ}_${SPEC%%:*}; MODE=${SPEC##*:}
    N=$((N+1))
    [ -s "out/TEXEL/${TAG}.json" ] && continue
    grep -qx "tex_${TAG}" /tmp/_wt.$$ && continue
    R=$(resolve "$OBJ" "$MODE")
    [ -z "$R" ] && { MISS="$MISS ${OBJ}/${MODE}"; continue; }
    if [ -n "${DRY:-}" ]; then printf '   %-26s %-5s %s\n' "$OBJ" "${SPEC%%:*}" "$(basename $R)"; continue; fi
    EXP="RUN=${R},TAG=${TAG},NFR=${NFR},SCRIPT=render_rung27_orbit.py"
    J=$(sbatch --parsable --job-name="tex_${TAG}" --export=ALL,"$EXP" jobs/texel_any.sbatch)
    note "$J" "tex_${TAG}" "$EXP" "jobs/texel_any.sbatch" \
      "WINDOW ablation texel: ${OBJ} ${MODE} (${SPEC%%:*}), ${NFR}fr -> out/TEXEL/${TAG}.json; PASS: json exists"
    echo "  $J  tex_${TAG}"
    S=$((S+1))
  done
done < "$TSV"
rm -f /tmp/_wt.$$
echo; echo "cells considered ${N}; submitted ${S}"
[ -n "$MISS" ] && echo "UNRESOLVED (no unique 30-epoch run with final_eval):$MISS"
exit 0
