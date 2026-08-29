#!/bin/bash
# bob_targets_and_train.sh -- steps 2 and 3 for the bob assets, GATED on step 1.
#
#   step 1  solve_orientation.py            (job 2224349) -> out/orient_bob/orientation.json
#   step 2  build_targets_new_objects.py    bakes canonical(mesh) @ R into
#                                           <obj>_render_frame.obj, make_render_mask --raw,
#                                           then the 2D copy
#   step 3  fig27.sbatch MODE=v2_D          rung27 qkvo+sa r4 + MCFM temporal-only
#
# THE GATES. Step 2 destroys nothing but writes 150 targets per object, and a target set
# built on a wrong pose fails visibly only days later. So this refuses to run unless:
#   * the spot_lava CONTROL comes back PASS at IoU >= 0.95. It is already aligned, so a
#     search that cannot recover identity on it is broken and no other row is readable.
#   * every bob row is verdict PASS.
#   * no bob row is flagged ambiguous. bob is a near-symmetric torus, so front/back is a
#     live risk; silently taking one of two equal scores is how a mesh trains back-to-front.
#   * the two bob solves agree. Both assets share ONE mesh and ONE conditioning render, so
#     their rotations must match -- an independent consistency check, the same one batch E
#     used when two assets shared a mesh.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
ORIENT=$E/out/orient_bob/orientation.json
OBJS="bob_spots,bob_spots_slow"
cd "$E"

python3 - "$ORIENT" <<'PY'
import json, sys, numpy as np
rows = json.load(open(sys.argv[1]))
by = {r['object']: r for r in rows}
ctl = next((r for r in rows if 'CONTROL' in r['object']), None)
assert ctl, 'no CONTROL row -- refusing'
print(f"  CONTROL {ctl['object']:22s} verdict={ctl['verdict']} iou={ctl['iou_after']:.4f}")
assert ctl['verdict'] == 'PASS' and ctl['iou_after'] >= 0.95, \
    'GATE-solve FAILED: the control did not recover identity; the search is broken'
bobs = [r for r in rows if r['object'].startswith('bob_')]
assert len(bobs) == 2, f'expected 2 bob rows, got {len(bobs)}'
for r in bobs:
    print(f"  {r['object']:30s} verdict={r['verdict']} iou={r['iou_after']:.4f} "
          f"ambiguous={r['ambiguous']} coarse={r['coarse_best']:.4f}/{r['coarse_runner_up']:.4f}")
    assert r['verdict'] == 'PASS', f"{r['object']} did not PASS"
    assert not r['ambiguous'], f"{r['object']} is AMBIGUOUS -- eyeball the contact sheet first"
# Compare the two solves as an ANGLE, not as a matrix-entry difference. The refine is
# coordinate descent at 1 deg, so two independent solves of the same pose land up to a
# step or two apart by construction: 1 deg of rotation is already max|dR| ~ 0.0175. An
# entrywise threshold of 1e-3 is 0.057 deg -- an order of magnitude below what the search
# can even resolve, so it rejects agreement. 3 deg is three refine steps.
R1 = np.array(bobs[0]['R'], float); R2 = np.array(bobs[1]['R'], float)
ang = np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
print(f"  two solves agree to {ang:.3f} deg (refine grid 1 deg), "
      f"IoU {bobs[0]['iou_after']:.4f} vs {bobs[1]['iou_after']:.4f}")
assert ang < 3.0, f'the two bob solves differ by {ang:.2f} deg, yet they share one mesh and one render'
print('  ALL GATES PASS')
PY

echo; echo "== step 2: bake pose + 2D copy =="
${PY2} -u "$E/build_targets_new_objects.py" --orient "$ORIENT" --objects "$OBJS" \
  --res 960 2>&1 | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo; echo "== step 3: rung27 + MCFM v2_D =="
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
for OBJ in ${OBJS//,/ }; do
  MESH="$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj"
  GTDIR="$E/out/gt_targets_$OBJ/frames"
  NFR=$( (find "$T2/data/$OBJ/frames_from_video" -maxdepth 1 -name 'frame_*.png' 2>/dev/null || true) | wc -l )
  [ -f "$MESH" ]  || { echo "MISSING posed mesh $MESH"; exit 1; }
  [ -d "$GTDIR" ] || { echo "MISSING targets $GTDIR"; exit 1; }
  JN="27_v2_D_${OBJ}"
  EXP="OBJ=${OBJ},MODE=v2_D,NFR=${NFR},MESH=${MESH},GTDIR=${GTDIR}"
  J=$(sbatch --parsable --job-name="$JN" --export=ALL,"$EXP" jobs/fig27.sbatch)
  note "$J" "$JN" "$EXP" "jobs/fig27.sbatch" \
    "BOB continuation: rung27 qkvo+sa r4 + MCFM v2_D, ${NFR}fr, ${OBJ}; PASS: log has '[FINAL] rung' and the run's final_eval.json exists"
  echo "  $J  $JN  (${NFR} frames)"
done
echo "done"
