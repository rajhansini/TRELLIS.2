#!/bin/bash
# submit_window_ablation.sh -- the W = 1 / 3 / 5 window ablation on spot_lava,
# plus the fixed-view export that the video-space comparison metrics need.
#
# WHY spot_lava. PAPER_STATUS_3DV27.md names it as the ablation figure object, it is
# the system-figure object, and it is the only figure-roster object inside the
# 24-object evaluation set.
#
# THE WINDOW IS THE MCFM CONDITIONING WINDOW, NOT rung32's --context-window.
# rung32 is a different operator (wide-context LoRA on the attention stack) and all
# 24 of its runs are context_window=3 anyway, so it cannot supply this curve.
#
# TEMPORAL-ONLY VARIANT (v2 = temporal_only). Token i attends over token i in
# neighbouring frames only. v3 pools space and time into one softmax and is a
# different question.
#
# RUN HASHES ARE PINNED, NOT GLOBBED. spot_lava has THREE rung27 no-mcfm runs
# (23.994 / 19.731 / 11.807 dB) and TWO v2_D runs (24.194 / 20.210). Globbing would
# pick one at random and put a cell in the table that is 4 dB from its neighbours.
# The three pinned below differ from each other in `mcfm` and NOTHING ELSE --
# verified field by field against their config.json.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
OBJ=spot_lava
NFR=150
MESH=$T2/data/spot_lava/mesh/spot_render_frame.obj
GTDIR=$E/out/gt_targets_spot_lava/frames
W1=runs/rung27_l1_lp_all_qkvo+sa_r4_s42_e2413d94          # mcfm=None   PSNR 23.994
W2=runs/rung27_l1_lp_mcfmv2_C_all_qkvo+sa_r4_s42_0291d161 # mcfm=v2_C   PSNR 24.061
W3=runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_59c51f6e # mcfm=v2_D   PSNR 24.194

note() {   # jid name export script desc
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
  echo "  $1  $2"
}

echo "== W=5 training (temporal-only, 5-frame window) =="
EXP="OBJ=${OBJ},NFR=${NFR},MODE=v2_E,MESH=${MESH},GTDIR=${GTDIR}"
A=$(sbatch --parsable --job-name="m3_${OBJ}_v2_E" --constraint="a40|L40S" \
     --export=ALL,"$EXP" "$E/jobs/rung27_mcfm_mode.sbatch")
note "$A" "m3_${OBJ}_v2_E" "$EXP" "jobs/rung27_mcfm_mode.sbatch" \
  "WINDOW ABLATION W=5: rung27 + MCFM temporal-only 5-frame window (v2_E, offsets -2..+2), ${OBJ}, ${NFR}fr, seed42; v2_E added to mcfm_blend.py today (_OFFSETS['E']); identical to the W=1/W=3 runs in every config field except mcfm; out=$E/out/r27m3_${A}.log; PASS: log has '[FINAL] rung27' and runs/rung27_l1_lp_mcfmv2_E_all_qkvo+sa_r4_s42_*/final_eval.json exists"

echo "== texel metrics for W=1 / W=2 / W=3, run hashes pinned =="
for pair in "w1:$W1:none" "w2:$W2:v2_C" "w3:$W3:v2_D"; do
  W=${pair%%:*}; rest=${pair#*:}; RUN=${rest%%:*}; MODE=${rest##*:}
  TAG="${OBJ}_${W}"
  [ -s "$E/out/TEXEL/${TAG}.json" ] && { echo "  skip $TAG (already measured)"; continue; }
  EXP="RUN=${RUN},TAG=${TAG},NFR=${NFR},SCRIPT=render_rung27_orbit.py"
  J=$(sbatch --parsable --job-name="tex_${TAG}" --export=ALL,"$EXP" "$E/jobs/texel_any.sbatch")
  note "$J" "tex_${TAG}" "$EXP" "jobs/texel_any.sbatch" \
    "WINDOW ABLATION ${W} texel flicker/accel/drift, ${OBJ}, mcfm=${MODE}, run pinned to ${RUN##*_}; the renderer takes the blend from the run's own config (_MCFM = ARGS.mcfm or CFG['mcfm']) and asserts it is not a no-op; out=$E/out/tex_${J}.log; PASS: $E/out/TEXEL/${TAG}.json exists and is non-empty"
done

echo "== texel metrics for W=5, dependent on the training job =="
TAG="${OBJ}_w5"
EXP="OBJ=${OBJ},MODE=v2_E,TAG=${TAG},NFR=${NFR},RUNG=27"
Eb=$(sbatch --parsable --job-name="texdep_${TAG}" --dependency=afterok:$A \
      --export=ALL,"$EXP" "$E/jobs/texel_after_train.sbatch")
note "$Eb" "texdep_${TAG}" "$EXP" "jobs/texel_after_train.sbatch" \
  "WINDOW ABLATION W=5 texel metrics, waits on afterok:${A}; resolves the run dir from (rung27, v2_E, ${OBJ}) via config.json rather than by newest-dir, and exits non-zero unless exactly one match has a checkpoint; out=$E/out/texdep_${Eb}.log; PASS: $E/out/TEXEL/${TAG}.json exists and is non-empty"

echo "== fixed-view frames for the video-space comparison metrics =="
TAG="${OBJ}_w3"
EXP="RUN=${W3},TAG=${TAG},NFR=${NFR},RES=518"
F=$(sbatch --parsable --job-name="fxv_${TAG}" --export=ALL,"$EXP" "$E/jobs/fixview_arm.sbatch")
note "$F" "fxv_${TAG}" "$EXP" "jobs/fixview_arm.sbatch" \
  "FIXED-VIEW frames (frozen | ours composite, --turns 0) for video_metrics.py, ${OBJ} rung27+v2_D, ${NFR}fr at 518px to match out/frozen_t2/renders which is 518; out=$E/out/fxv_${F}.log; PASS: $E/out/fixview_${TAG}/frames holds ${NFR} pngs"

echo; echo "submitted: train=$A  texel(w1,w2,w3)  texdep=$Eb  fixview=$F"
