#!/bin/bash
# submit_continuation_dailies.sh -- everything the dailies pages need for the
# textured-continuation objects, matching what batches C/D/E already have.
#
# A daily is a FOUR-column panel: GT | frozen | rung27 | rung27+MCFM. It is built from
# two renders, each a frozen|adapted pair:
#     view_<obj>_27_<view>    rung27, MCFM off
#     view_<obj>_27m_<view>   rung27 + MCFM v2_D
# at train (yaw 0, elev 0) and the three unseen diagonals A/B/C.
#
# THE GAP THIS FILLS. The continuation objects were trained on the MCFM arm ONLY, so
# the '27' column does not exist yet. This script trains it, then chains the renders
# with --dependency=afterok so nothing renders against a missing checkpoint.
#
# Run dirs are resolved from config.json by exact '/data/<obj>/' segment match, never
# globbed by name -- the same rule submit_dailies_de.sh follows.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"
OBJS="${OBJS:-plane_waves_from_frame_150 chair_real_wooden_crack_from_frame_150}"
# Which arms to build, as "<tag>:<config mcfm>:<train MODE>" tokens. Default is both
# columns of the four-column daily. Override to a single arm when the plain-rung27
# column is not wanted -- otherwise the 27 arm has no run, and this script would
# submit a 3h30 fig27 training per object just to fill a column nobody asked for.
#     ARMS="27m:v2_D:v2_D"   MCFM arm only  -> three-column GT | frozen | rung27+MCFM
ARMS="${ARMS:-27:None: 27m:v2_D:v2_D}"

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

resolve() {  # obj mcfm(None|v2_D) -> run dir, or empty
  python3 - "$E" "$1" "$2" <<'PY'
import json,sys,glob,os
E,obj,mode=sys.argv[1:4]
hit=[]
for c in glob.glob(f'{E}/runs/rung27_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if str(d.get('mcfm'))!=mode: continue
    if d.get('rung')!=27 or d.get('epochs')!=30: continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if not os.path.exists(os.path.join(os.path.dirname(c),'ckpts','lora_best.pt')): continue
    hit.append(os.path.dirname(c))
print(hit[0] if len(hit)==1 else '')
PY
}

# job id of a still-live training job for <obj>/<mode>, so renders can chain on it
livejob() { squeue -u "$USER" -h -o "%i %j" | awk -v n="$1" '$2==n{print $1; exit}'; }

VIEWS="train:0.0:0.0 diagA:45.0:25.0 diagB:135.0:-20.0 diagC:225.0:30.0"
SUB=0
for OBJ in $OBJS; do
  MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
  GTDIR="$E/out/gt_targets_$OBJ/frames"
  NFR=$( (find "$T2/data/$OBJ/frames_from_video" -maxdepth 1 -name 'frame_*.png' 2>/dev/null || true) | wc -l )
  echo "== $OBJ  (nfr=$NFR)"

  for SPEC in $ARMS; do
    TAG=${SPEC%%:*}; REST=${SPEC#*:}; CFGMODE=${REST%%:*}; MODE=${REST##*:}
    RUN=$(resolve "$OBJ" "$CFGMODE")
    DEP=""

    if [ -z "$RUN" ]; then
      # no finished run. Either one is already training, or we must start it.
      TN="27${MODE:+_$MODE}_${OBJ}"
      LJ=$(livejob "$TN")
      if [ -n "$LJ" ]; then
        echo "   $TAG: training live as $LJ ($TN) -- chaining renders on it"
        DEP="$LJ"
      else
        EXP="OBJ=${OBJ},MODE=${MODE},NFR=${NFR},MESH=${MESH},GTDIR=${GTDIR}"
        LJ=$(sbatch --parsable --job-name="$TN" --export=ALL,"$EXP" jobs/fig27.sbatch)
        note "$LJ" "$TN" "$EXP" "jobs/fig27.sbatch" \
          "CONTINUATION dailies: rung27 qkvo+sa r4${MODE:+ mcfm=$MODE}, ${NFR}fr, ${OBJ}; PASS: log has '[FINAL] rung' (log out/FIGRUNS/${TN}_${LJ}.log)"
        echo "   $LJ  $TN   (training the missing $TAG arm)"
        DEP="$LJ"; SUB=$((SUB+1))
      fi
      # renders must resolve RUN at run time -- it does not exist yet. render_arm.sbatch
      # needs RUN=, so the render is submitted by a follow-up pass once training lands.
      echo "   $TAG: renders deferred until $DEP finishes (re-run this script then)"
      continue
    fi

    for V in $VIEWS; do
      VIEW=${V%%:*}; R=${V#*:}; YAW=${R%%:*}; ELEV=${R##*:}
      OUTD="$E/out/view_${OBJ}_${TAG}_${VIEW}"
      # A run dir with ckpts/lora_best.pt is NOT proof training finished -- that file is
      # rewritten every time an epoch improves. Rendering a live run would silently use a
      # mid-training checkpoint. Chain on the live job instead.
      LIVE=$(livejob "27${MODE:+_$MODE}_${OBJ}")
      DEPFLAG=""; [ -n "$LIVE" ] && DEPFLAG="--dependency=afterok:$LIVE"
      have=$( (find "$OUTD/frames" -maxdepth 1 -name '*.png' 2>/dev/null || true) | wc -l )
      [ "$have" -ge "$NFR" ] && { echo "   skip $TAG/$VIEW ($have frames)"; continue; }
      squeue -u "$USER" -h -o "%j" | grep -qx "dl_${OBJ}_${TAG}_${VIEW}" && { echo "   queued $TAG/$VIEW"; continue; }
      EXP="RUN=${RUN},TAG=view_${OBJ}_${TAG}_${VIEW},YAW0=${YAW},ELEV=${ELEV},TURNS=0,NFR=${NFR},RES=518"
      J=$(sbatch --parsable --job-name="dl_${OBJ}_${TAG}_${VIEW}" $DEPFLAG --export=ALL,"$EXP" jobs/render_arm.sbatch)
      note "$J" "dl_${OBJ}_${TAG}_${VIEW}" "$EXP" "jobs/render_arm.sbatch" \
        "CONTINUATION dailies: ${OBJ} ${TAG} at ${VIEW} (yaw ${YAW}, elev ${ELEV}), ${NFR}fr, frozen|adapted -> out/view_${OBJ}_${TAG}_${VIEW}/; PASS: ${NFR} pngs in frames/"
      echo "   $J  dl_${OBJ}_${TAG}_${VIEW}"
      SUB=$((SUB+1))
    done
  done
done
echo; echo "submitted $SUB job(s)"
