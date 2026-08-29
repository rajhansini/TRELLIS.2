#!/bin/bash
# submit_texel_r19.sh — texel temporal metrics (flicker / accel / drift) for the
# rung19 CROSS-ATTENTION-ONLY arm, row (b) of the ablation table.
#
# WHY THIS EXISTS
#   The fig19 batch trained 22 objects overnight and every one wrote final_eval.json,
#   so row (b) has PSNR and SSIM. It has no temporal numbers: flicker/accel/drift come
#   from a separate ODE decode over the PBR voxel field, which training never runs.
#   Without this pass the row cannot go into results_table.tex at all.
#
# SCRIPT DISPATCH IS MEASURED, NOT GUESSED (see the header of texel_any.sbatch).
#   Verified from runs/rung19_*/ckpts/lora_best.pt: 180 tensors, 0 gates -- the six
#   cross-attention tensors per block over 30 blocks, and no dual-branch gates. That
#   is neither the 300-tensor rung27 family nor the 510-tensor rung31 family, so
#   neither renderer's *default* registry matches it. render_rung27_orbit.py is
#   nonetheless correct because it builds the registry from the RUN'S OWN
#   config.json (`targets=tuple(CFG['target_set'])`, rank and alpha likewise) rather
#   than from a constant, so it reconstructs the 180-tensor shape exactly and
#   load_state_dict is strict-clean. render_rung31_orbit.py would add the gated
#   branch and raise on missing keys.
#
# ONE JOB PER OBJECT, never a loop inside one job.
#
# RUN AND NFR ARE PARAMETERS, READ FROM EACH OBJECT'S OWN CONFIG.
#   out/texel_r19_params.tsv is generated from runs/*/config.json. Hardcoding 150
#   would silently measure the 121-frame objects (pumpkin_rot) over frames they do
#   not have. The frame count is per-object and comes from the run that produced
#   the checkpoint.
#
# THE OBJECT SET IS THE TABLE'S 24, NOT THE FIG19 BATCH'S 22.
#   pumpkin_rot and spot_lava were skipped by submit_fig19_row_b.sh because they
#   already had complete rung19 runs (3bb4375e, 60c8897c). They are still rows in
#   results_table.tex, so they are measured here or the row averages over a
#   different object set than the rows above it.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
PARAMS=$E/out/texel_r19_params.tsv
SCRIPT=render_rung27_orbit.py
mkdir -p "$E/out/TEXEL"
[ -f "$PARAMS" ] || { echo "missing $PARAMS"; exit 1; }

n=0; skipped=0
while IFS=$'\t' read -r OBJ RUN NFR; do
  [ "$OBJ" = "object" ] && continue
  [ -z "$OBJ" ] && continue
  TAG="${OBJ}_r19"
  OUTJ="$E/out/TEXEL/${TAG}.json"
  if [ -s "$OUTJ" ]; then
    echo "skip  $TAG  (already measured)"; skipped=$((skipped+1)); continue
  fi
  [ -f "$E/$RUN/config.json" ] || { echo "NO config.json under $RUN -- skipping $OBJ"; continue; }
  JN="tex_${TAG}"
  EXP="RUN=${RUN},TAG=${TAG},NFR=${NFR},SCRIPT=${SCRIPT}"
  JID=$(sbatch --parsable --job-name="$JN" --export=ALL,"$EXP" "$E/jobs/texel_any.sbatch")
  python3 - "$MAN" "$JID" "$JN" "$EXP" <<'PY'
import json,sys,os
man,jid,name,exp=sys.argv[1:5]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":"jobs/texel_any.sbatch","name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  /net/projects/ranalab/rajhansini/joblog.sh add "$JID" "tex_${TAG}" \
    "TEXEL flicker/accel/drift, rung19 CA-only arm (row b of ablation table), ${NFR}fr, ${OBJ}; renderer=${SCRIPT} (registry rebuilt from run config, 180 tensors 0 gates); out=$E/out/tex_${JID}.log; PASS: $E/out/TEXEL/${TAG}.json exists and is non-empty"
  echo "$JID  $JN  nfr=$NFR  run=$RUN"
  n=$((n+1))
done < "$PARAMS"
echo "submitted $n jobs, skipped $skipped already-measured"
