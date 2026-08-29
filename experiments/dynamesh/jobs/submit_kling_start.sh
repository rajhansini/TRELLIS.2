#!/bin/bash
# submit_kling_start.sh -- fixed-view renders of the rung27 + MCFM v2_D arm for the
# two batch-B figure objects, so their LAST frame can be used as the Kling start
# frame for the textured-continuation experiment.
#
# WHY THESE TWO AND NOT ALL FOUR
#   hand_rorschach and pumpkin_rot already have out/fixview_<obj>_mcfm_yaw0/frames
#   on disk from the video-comparison pass. chair_real_wooden_crack and
#   napolean_waves only ever went through jobs/texel_metric.sbatch, which passes
#   --skip-render, so their frames dirs exist but are empty. Nothing is recomputed
#   here that already exists.
#
# WHY --turns 0
#   The camera must be pinned. An orbit render changes the image between frames
#   whether or not the texture moved, and the start frame has to be the training
#   view so the continuation video lands in the pose the mesh was fitted to.
#
# RES=960 matches build_targets_hero.py --res 960, the resolution the GT targets
# were built at, and gives Kling a 960x960 start frame after the panel crop.
#
# One job per object -- never a loop inside one job.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
cd "$E"

note() {   # note <jid> <jobname> <export-string> <script> <passcond>
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
man, jid, name, exp, script = sys.argv[1:6]
m = json.load(open(man)) if os.path.exists(man) else {}
m[jid] = {"script": script, "name": name, "export": exp}
json.dump(m, open(man, "w"), indent=1)
PY
  $JL add "$1" "$2" "$5"
}

RES=960
SUB=0
while IFS=$'\t' read -r OBJ RUN NFR; do
  TAG="${OBJ}_mcfm_yaw0"
  FR="$E/out/fixview_${TAG}/frames"
  # idempotent: a complete render is never redone
  n=$( (find "$FR" -maxdepth 1 -name '*.png' 2>/dev/null || true) | wc -l )
  if [ "$n" -eq "$NFR" ]; then echo "SKIP $OBJ -- $n/$NFR frames already on disk"; continue; fi
  [ -f "$E/$RUN/config.json" ] || { echo "MISSING config.json under $RUN"; exit 1; }
  NAME="fxv_${OBJ}_kstart"
  EXP="RUN=${RUN},TAG=${TAG},NFR=${NFR},RES=${RES}"
  JID=$(sbatch --parsable --job-name="$NAME" --export=ALL,"$EXP" jobs/fixview_arm.sbatch)
  note "$JID" "$NAME" "$EXP" "jobs/fixview_arm.sbatch" \
       "out=$E/out/fixview_${TAG}/frames/ pass=${NFR} png written (log out/fxv_${JID}.log)"
  echo "  $JID  $NAME  -> out/fixview_${TAG}/frames  (${NFR} frames @ ${RES}px)"
  SUB=$((SUB+1))
done <<TSV
chair_real_wooden_crack	runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_55e4f367	150
napolean_waves	runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_301f5e50	150
TSV

echo "submitted $SUB job(s)"
