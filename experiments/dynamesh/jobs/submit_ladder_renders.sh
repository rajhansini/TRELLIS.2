#!/bin/bash
# submit_ladder_renders.sh -- pixel-space renders for the ablation ladder arms.
#
# Unlocks the VIDEO-SPACE component table Itai asked for: the texel ladder already
# exists, but the pixel table needs view_<obj>_<arm>_<view> renders for the arms
# that only ever had texel measurements.
#
#   19  rung19  cross-attention LoRA only        (mcfm None)
#   37  rung37  + learned per-position temporal  (mcfm None)
#   31  rung31  + joint spatio-temporal          (mcfm None)
#
# RUN RESOLUTION IS NOT A GLOB. rung19 has 29 objects with more than one 30-epoch
# run (some mcfm=None, some mcfm=v2_D), so a glob would silently pick the wrong
# checkpoint and the pixel row would not correspond to the published texel row.
# Instead each arm's run is read from the SAME out/TEXEL/<obj>_<arm>.json the
# published ladder used, so texel and pixel rows are the same checkpoint by
# construction. Verified: all 42 r19 runs are rung19/30ep/mcfm=None with a ckpt.
#
# rung31 covers only 24 of the 42 objects -- the other 18 were never trained, so
# they are reported as UNTRAINED rather than rendered. Rendering cannot invent them.
#
# Idempotent on the frames directory, so a rerun submits only what is missing.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
JL=/net/projects/ranalab/rajhansini/joblog.sh
MAN=$E/out/job_manifest.json
cd "$E"

ARMS=${ARMS:-"19 37 31"}
OBJS=$(python3 -c "import json;print(' '.join(json.load(open('out/fullrate_table_batchALL.json'))['objects']))")
squeue -u "$USER" -h -o "%j" > /tmp/_lr.$$ || true

N=0; S=0; SKIP=0; NORUN=""
for ARM in $ARMS; do
  case "$ARM" in 19) TEX=r19 ;; 37) TEX=r37 ;; 31) TEX=r31 ;; *) echo "unknown arm $ARM"; exit 1 ;; esac
  for OBJ in $OBJS; do
    # the run the published texel ladder used for this (object, arm)
    RUN=$(python3 - "$E" "$OBJ" "$TEX" <<'PY'
import json,os,sys
E,obj,tex=sys.argv[1:4]
p=f'{E}/out/TEXEL/{obj}_{tex}.json'
if not os.path.exists(p): print(''); raise SystemExit
r=json.load(open(p))['run'].split('/')[-1]
d=f'{E}/runs/{r}'
print(d if os.path.exists(f'{d}/ckpts/lora_best.pt') and os.path.exists(f'{d}/config.json') else '')
PY
)
    [ -z "$RUN" ] && { NORUN="$NORUN ${OBJ}/${ARM}"; continue; }
    # rung37_objects.tsv columns are: batch, object, n_frames, mesh, gt_dir, gt_render_dir.
    # Matching $1 matches the BATCH LETTER, never the object, so every lookup missed and
    # NFR silently fell back to 150. pumpkin_rot has 121 frames, so its renders asked for
    # frame_0122.png and died with FileNotFoundError -- 12 jobs, one per view per arm.
    # The object is $2.
    NFR=$(awk -F"\t" -v o="$OBJ" '$2==o{print $3}' "$E/jobs/rung37_objects.tsv"); NFR=${NFR:-150}
    for V in "train:0.0:0.0" "diagA:45.0:25.0" "diagB:135.0:-20.0" "diagC:225.0:30.0"; do
      VIEW=${V%%:*}; R=${V#*:}; YAW=${R%%:*}; ELEV=${R##*:}
      N=$((N+1))
      TAG="view_${OBJ}_${ARM}_${VIEW}"
      # NEVER `ls glob | wc -l` under `set -eo pipefail`: a missing frames dir makes
      # ls exit 2, pipefail propagates it to the assignment, and set -e kills the
      # script WITHOUT PRINTING ANYTHING. This is the identical trap render_arm.sbatch
      # documents in its own header, and here it silently truncated the run at the
      # first object that had no frames directory yet -- i.e. exactly the objects
      # this script exists to render.
      C=0
      [ -d "out/${TAG}/frames" ] && C=$(find "out/${TAG}/frames" -maxdepth 1 -name '*.png' | wc -l)
      [ "$C" -ge "$NFR" ] && { SKIP=$((SKIP+1)); continue; }
      grep -qx "a${ARM}_${OBJ}_${VIEW}" /tmp/_lr.$$ && { SKIP=$((SKIP+1)); continue; }
      # jobs/submit_arm_renders.sh renders the SAME (object, arm, view) cells into the
      # SAME out/view_<obj>_<arm>_<view>/frames, under the names r19v_/r37v_/r31v_.
      # Two renderers writing one frames dir at once can tear a PNG mid-save, so a
      # cell already queued there is skipped here rather than raced.
      case "$ARM" in 31) MINE="r31v" ;; *) MINE="r${ARM}v" ;; esac
      grep -qx "${MINE}_${OBJ}_${VIEW}" /tmp/_lr.$$ && { SKIP=$((SKIP+1)); continue; }
      [ -n "${DRY:-}" ] && { printf '   %-26s arm=%-3s %-6s nfr=%s\n' "$OBJ" "$ARM" "$VIEW" "$NFR"; continue; }
      EXP="RUN=${RUN},TAG=${TAG},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=${NFR},RES=518"
      J=$(sbatch --parsable --job-name="a${ARM}_${OBJ}_${VIEW}" --export=ALL,"$EXP" jobs/render_arm.sbatch)
      python3 - "$MAN" "$J" "a${ARM}_${OBJ}_${VIEW}" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/render_arm.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
      $JL add "$J" "a${ARM}_${OBJ}_${VIEW}" \
        "ladder render: ${OBJ} arm=${ARM} view=${VIEW} ${NFR}fr -> out/${TAG}/frames; PASS: ${NFR} pngs"
      S=$((S+1))
    done
  done
done
rm -f /tmp/_lr.$$
echo "cells ${N}; submitted ${S}; already done/queued ${SKIP}"
[ -n "$NORUN" ] && { echo "UNTRAINED (no 30-epoch checkpoint -- needs training, not rendering):"; for x in $NORUN; do echo "   $x"; done; }
exit 0
