#!/bin/bash
# submit_window_timing.sh -- does the MCFM window cost anything at inference?
#
# THE QUESTION
#   timing_table.tex already asserts "the temporal component is parameter-free and
#   adds no measurable cost", on the reasoning that the blend is applied once to the
#   cached conditioning tokens before the flow runs. That is an argument, not a
#   measurement. This measures it.
#
# THE ARM
#   One adapter, hand_rorschach, applied zero-shot to a mesh it never saw (unicorn),
#   at each of the seven windows in the window-ablation table. Same mesh, same
#   conditioning, same noise, same views, same frame count for every row, so the only
#   thing that varies is the window.
#
#   W=1 is the arm with NO blend at all, and its adapter was trained without MCFM.
#   MCFM= is therefore deliberately EMPTY there. Passing --mcfm anyway would trip
#   the renderer's GATE-mcfm train/test-mismatch check, which is the correct
#   behaviour and not something to work around.
#
# BATCHING
#   BATCHES="1 8" runs each window twice: sequential, then frame-batched. Frame
#   batching is legal here because every frame shares one noise and one sparse
#   structure S and no frame reads another's output, so N frames go into the batch
#   dim under a single 12-step ODE. Running both arms per window separates "does the
#   window cost anything" from "does batching still help at a wide window".
#
# ONE JOB PER WINDOW. Never a loop inside one sbatch: a single failure would take
# the whole sweep with it, and seven independent jobs schedule in parallel.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=${T2}/experiments/dynamesh
MESH=${MESH:-${T2}/data/unicorn_rainbow/mesh/unicorn_rainbow_render_frame.obj}
COND=${COND:-${T2}/data/hand_rorschach/frames_from_video}
BATCHES=${BATCHES:-1 8}
NF=${NF:-150}
MANIFEST=${E}/out/job_manifest.json

# window -> "run_dir_basename mcfm_mode".  Empty mode = no blend (W=1).
declare -A ARM=(
  [1]="rung27_l1_lp_all_qkvo+sa_r4_s42_1eef6f88 "
  [3]="rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_5f95157f v2_D"
  [5]="rung27_l1_lp_mcfmv2_E_all_qkvo+sa_r4_s42_0000053a v2_E"
  [7]="rung27_l1_lp_mcfmv2_F_all_qkvo+sa_r4_s42_bcd459d3 v2_F"
  [11]="rung27_l1_lp_mcfmv2_G_all_qkvo+sa_r4_s42_b73875d4 v2_G"
  [13]="rung27_l1_lp_mcfmv2_H_all_qkvo+sa_r4_s42_5b452c07 v2_H"
  [15]="rung27_l1_lp_mcfmv2_I_all_qkvo+sa_r4_s42_4363700a v2_I"
)

[ -f "$MESH" ] || { echo "no mesh: $MESH" >&2; exit 1; }
[ -d "$COND" ] || { echo "no cond dir: $COND" >&2; exit 1; }

# Preflight every checkpoint BEFORE submitting any job. A missing lora_best.pt only
# surfaces after the GPU is allocated and the model is loaded, minutes in.
for W in 1 3 5 7 11 13 15; do
  read -r RUNDIR MODE <<< "${ARM[$W]}"
  CK="${E}/runs/${RUNDIR}/ckpts/lora_best.pt"
  [ -f "$CK" ] || { echo "W=$W missing checkpoint: $CK" >&2; exit 1; }
done
echo "preflight OK: 7 checkpoints, mesh, cond dir"

for W in 1 3 5 7 11 13 15; do
  read -r RUNDIR MODE <<< "${ARM[$W]}"
  NAME="benchpar_uni_w${W}"          # benchpar_ is already in watchdog.py's KIND table
  TAG="uni_w${W}"
  EXPORT="RUN=${E}/runs/${RUNDIR},MESH=${MESH},COND=${COND},MCFM=${MODE},TAGBASE=${TAG},NF=${NF},BATCHES=${BATCHES}"

  if [ -n "${DRY:-}" ]; then echo "[dry] ${NAME}  mcfm='${MODE}'"; continue; fi

  JID=$(sbatch --parsable --job-name="${NAME}" \
        --export=ALL,"${EXPORT}" "${E}/jobs/bench_parallel.sbatch")
  echo "  ${NAME}  jobid=${JID}  mcfm='${MODE:-none}'"

  # Manifest entry, written at submit time. watchdog.py replays exactly this export
  # on a requeue; without it a resubmitted job would fall back to the sbatch's
  # defaults and silently benchmark the wrong window on the wrong mesh.
  python3 - "$MANIFEST" "$JID" "$NAME" "$EXPORT" "${E}/jobs/bench_parallel.sbatch" <<'PY'
import json, sys, pathlib
mf, jid, name, exp, script = sys.argv[1:6]
p = pathlib.Path(mf)
d = json.loads(p.read_text()) if p.exists() else {}
d[jid] = {'script': script, 'name': name, 'export': exp}
p.write_text(json.dumps(d))
PY
  /net/projects/ranalab/rajhansini/joblog.sh add "${JID}" "${NAME}" \
    "window timing, hand_rorschach LoRA -> unicorn, W=${W} mcfm=${MODE:-none} batches='${BATCHES}'" >/dev/null
done

echo
echo "results land in ${E}/out/timing_uni_w<W>_b<BATCH>.json"
echo "aggregate with: python3 ${E}/jobs/report_window_timing.py"
