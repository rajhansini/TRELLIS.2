#!/bin/bash
# submit_continuation.sh -- the textured-continuation pipeline for one object.
#
#   step 1  jobs/targets_hero.sbatch   build the 2D copy from the Kling video onto our
#                                      mesh. Carries GATE-align, which DIES if our
#                                      silhouette is off the video's object.
#   step 2  jobs/fig27.sbatch          rung27 (qkvo+sa, rank 4) + MCFM v2_D, 30 epochs.
#
# Step 2 is submitted with --dependency=afterok on step 1, so if GATE-align fails the
# training never starts. That is the whole point of chaining rather than firing both:
# a continuation video whose object drifted off our mesh must not silently train.
#
# One job per object. OBJ is a parameter; nothing here is hardcoded to one asset.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"

OBJ=${OBJ:?set OBJ=<asset name>}
MODE=${MODE:-v2_D}

MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
GTDIR="$E/out/gt_targets_$OBJ/frames"
NFR=$( (find "$T2/data/$OBJ/frames_from_video" -maxdepth 1 -name 'frame_*.png' 2>/dev/null || true) | wc -l )

[ -f "$MESH" ] || { echo "MISSING mesh: $MESH"; exit 1; }
[ "$NFR" -ge 1 ] || { echo "MISSING frames under $T2/data/$OBJ/frames_from_video"; exit 1; }
echo "OBJ=$OBJ  MODE=$MODE  NFR=$NFR"
echo "MESH=$MESH"

note() {  # note <jid> <name> <export> <script> <passcond>
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
man, jid, name, exp, script = sys.argv[1:6]
m = json.load(open(man)) if os.path.exists(man) else {}
m[jid] = {"script": script, "name": name, "export": exp}
json.dump(m, open(man, "w"), indent=1)
PY
  $JL add "$1" "$2" "$5"
}

# ---- step 1: 2D copy + GATE-align -----------------------------------------
N1="tgt_${OBJ}"
E1="OBJ=${OBJ}"
J1=$(sbatch --parsable --job-name="$N1" --export=ALL,"$E1" jobs/targets_hero.sbatch)
note "$J1" "$N1" "$E1" "jobs/targets_hero.sbatch" \
     "2D copy for $OBJ; out=$E/out/gt_targets_${OBJ}/; PASS: gt_targets.json exists AND log has '[GATE-align]' without FAILED (log out/tgt_${J1}.log)"
echo "  $J1  $N1   -> out/gt_targets_${OBJ}/"

# ---- step 2: rung27 + MCFM, only if step 1 succeeded -----------------------
N2="27_${MODE}_${OBJ}"
E2="OBJ=${OBJ},MODE=${MODE},NFR=${NFR},MESH=${MESH},GTDIR=${GTDIR}"
J2=$(sbatch --parsable --job-name="$N2" --dependency=afterok:"$J1" --export=ALL,"$E2" jobs/fig27.sbatch)
note "$J2" "$N2" "$E2" "jobs/fig27.sbatch" \
     "rung27 qkvo+sa r4 + MCFM ${MODE}, ${NFR}fr, depends afterok:${J1}; PASS: log has '[FINAL] rung' and runs/rung27_l1_lp_mcfm${MODE}_all_qkvo+sa_r4_s42_*/final_eval.json exists (log out/FIGRUNS/${N2}_${J2}.log)"
echo "  $J2  $N2   (afterok:$J1)"
echo "submitted 2 jobs"
