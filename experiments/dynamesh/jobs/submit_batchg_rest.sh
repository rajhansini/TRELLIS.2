#!/bin/bash
# submit_batchg_rest.sh -- the batch-G clips that have targets but were never trained:
# car_effect1, car_effect2, robot_rust_2. Same seven supplementary arms batch F and the
# two unicorns got, one job per (object, arm), 21 jobs. No target step: all three already
# have 150 gated frames in out/gt_targets_<obj>/frames.
#
# ALIGNMENT, recorded here because it is the reason these sat unsubmitted:
#   robot_rust_2  IoU 0.939  uncovered 2.00%   <- the regeneration; clears the 0.90 bar
#   car_effect1   IoU 0.898  uncovered 4.58%   <- scale-corrected, still 0.002 under 0.90
#   car_effect2   IoU 0.898  uncovered 4.41%   <- same
# The cars are a deliberate exception, not an oversight: they sit a hair under the bar
# that gargoyle (0.754) failed, and their targets were rebuilt with the similarity
# correction already applied. Train them and judge the renders.
#
#   ONLY=obj1,obj2   restrict objects
#   ARMS=r19,r27     restrict arms
#   DRY=1            print the work list, submit nothing
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="${OBJS:-car_effect1 car_effect2 robot_rust_2}"
ARMS="${ARMS:-r19,r19mcfm,r27,r27mcfm,w5,stD,v3D}"

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

squeue -u "$USER" -h -o "%j" > /tmp/_bg.$$ || true
N=0; S=0
for OBJ in $OBJS; do
  MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
  GTREN="$E/out/gt_targets_$OBJ/frames"
  NFR=$(find "$T2/data/$OBJ/frames_from_video" -maxdepth 1 -name 'frame_*.png' 2>/dev/null | wc -l)
  # Every guard that bit an earlier batch: the mesh, the BACK-PROJECTED targets (not the
  # raw video frames), and the mask GATE-align reads. A missing render_mask.npy is what
  # sent batch D/E into "GATE-align FAILED" after the GPU was already allocated.
  [ -f "$MESH" ]  || { echo "   MISSING MESH        $OBJ"; continue; }
  [ -d "$GTREN" ] || { echo "   MISSING TARGETS     $OBJ"; continue; }
  [ -f "$E/out/gt_targets_$OBJ/render_mask.npy" ] || { echo "   MISSING render_mask $OBJ"; continue; }
  [ "$NFR" -ge 30 ] || { echo "   ONLY $NFR FRAMES    $OBJ"; continue; }
  echo "== $OBJ  (nfr=$NFR)"

  # tag:job-name:sbatch:extra-export -- exactly what jobs 2236927-2236933 used.
  for SPEC in \
    "r19:19_${OBJ}:jobs/fig19.sbatch:" \
    "r19mcfm:r26ca_${OBJ}:jobs/rung26_mcfm_ca.sbatch:MODE=v2_D" \
    "r27:r27_${OBJ}:jobs/rung27_hero.sbatch:" \
    "r27mcfm:mcf_${OBJ}:jobs/rung27_mcfm_hero.sbatch:" \
    "w5:m3_v2_E_${OBJ}:jobs/rung27_mcfm_mode.sbatch:MODE=v2_E" \
    "stD:m3_st_D_${OBJ}:jobs/rung27_mcfm_mode.sbatch:MODE=st_D" \
    "v3D:m3_v3_D_${OBJ}:jobs/rung27_mcfm_mode.sbatch:MODE=v3_D" ; do
    TAG=${SPEC%%:*}; R=${SPEC#*:}; JN=${R%%:*}; R=${R#*:}; SB=${R%%:*}; EXTRA=${R#*:}
    [[ ",$ARMS," != *",$TAG,"* ]] && continue
    N=$((N+1))
    [ -f "out/TEXEL/${OBJ}_${TAG}.json" ] && { echo "   skip $TAG (texel exists)"; continue; }
    grep -qx "$JN" /tmp/_bg.$$ && { echo "   skip $TAG (queued)"; continue; }
    # rung27_hero / rung27_mcfm_hero resolve MESH and GTDIR from OBJ themselves; the
    # other three take them explicitly. Passing both is harmless and keeps one code path.
    EXP="OBJ=${OBJ},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN}${EXTRA:+,$EXTRA}"
    [ -n "${DRY:-}" ] && { printf '   %-8s %-28s %s\n' "$TAG" "$JN" "$SB"; continue; }
    J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" --export=ALL,"$EXP" "$SB")
    note "$J" "$JN" "$EXP" "$SB" \
      "BATCH G rest: ${OBJ} arm=${TAG} (${SB##*/}) ${NFR}fr seed42 30ep; PASS: out/TEXEL/${OBJ}_${TAG}.json after the texel pass, and the run dir has final_eval.json"
    echo "   $J  $JN  ($TAG)"
    S=$((S+1))
  done
done
rm -f /tmp/_bg.$$
echo; echo "cells considered ${N}; submitted ${S}"
