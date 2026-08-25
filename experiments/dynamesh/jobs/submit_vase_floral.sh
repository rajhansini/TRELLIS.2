#!/bin/bash
# submit_vase_floral.sh — vase_floral end to end: frames -> targets -> rung27+MCFM.
#
# The vase's pose is KNOWN, not searched: it was rendered for Kling at yaw 240 /
# pitch 25 (chosen by vase_sweep.py — 45.9% of the surface visible, handle 52.7%)
# and known_rotations/kling_rotations.json carries the matrices, verified byte-exact
# against the PNG that went to Kling. So there is no solve_orientation step here.
#
# THE CONVERSION IS  R_ours = F @ R_total @ M.T,  NOT  M @ R @ M.T.
# The provenance README suggests the similarity form; verify_known_rotations.py
# measured it and it lands 174 deg off, because our canonical map M and the camera
# frame change F differ by a 180 deg yaw. Getting this wrong yields a mesh facing
# away from the video with a plausible-looking silhouette.
#
# MODE=v2_D is temporal_only_w3: token j attends over token j in frames t-1,t,t+1.
# It is the mode that won on both trained objects and is rung27's standing default.
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OBJ=vase_floral
MP4=$T2/data/$OBJ/video/$OBJ.mp4
FRAMES=$T2/data/$OBJ/frames_from_video
ORIENT=out/orient/orientation_vase.json

note() {   # jid name export script desc
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
  echo "  submitted $1  $2"
}

# ---- 0. the video must actually be there -----------------------------------
[ -s "$MP4" ] || { echo "FATAL: $MP4 missing or empty. scp it first."; exit 1; }
echo "== video: $(du -h "$MP4" | cut -f1)  $MP4"

# ---- 1. frames -------------------------------------------------------------
# frame_%04d.png from 1, the naming every other object uses.
mkdir -p "$FRAMES"
rm -f "$FRAMES"/frame_*.png
# 960x960, NOT native. The kling mp4 is 1440^2 but all seven existing objects
# store 960^2 frames and the whole target/training path is proven at that size; matching
# them keeps this run on the tested path instead of introducing a new variable.
ffmpeg -nostdin -loglevel error -i "$MP4" -vf scale=960:960:flags=lanczos \
  -start_number 1 "$FRAMES/frame_%04d.png"
NFR=$(ls "$FRAMES"/frame_*.png | wc -l)
[ "$NFR" -ge 30 ] || { echo "FATAL: only $NFR frames extracted"; exit 1; }
echo "== frames: $NFR -> $FRAMES"

# ---- 2. orientation entry in OUR canonical frame ---------------------------
$PY2 - "$OBJ" "$NFR" <<'PY'
import json, sys, numpy as np
from pathlib import Path
obj, nfr = sys.argv[1], int(sys.argv[2])
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
K = json.load(open(E/'known_rotations/kling_rotations.json'))['objects']['vase']
M = np.array([[-1.,0,0],[0,0,1.],[0,1.,0]])      # canonical() axis map (-x, z, y)
F = np.array([[1.,0,0],[0,0,-1.],[0,1.,0]])      # his camera frame -> ours (x, -z, y)
R = F @ np.array(K['R_total']) @ M.T
assert abs(np.linalg.det(R)-1) < 1e-9 and np.allclose(R@R.T, np.eye(3), atol=1e-9), 'not SO(3)'
out = [{'object': obj,
        'iou_after': -1.0,               # sentinel: KNOWN rotation, nothing was solved
        'verdict': 'KNOWN_ROT',
        'source': 'kling_rotations.json[vase], verified_byte_exact; R_ours = F @ R_total @ M.T',
        'R': [[float(x) for x in r] for r in R],
        'mesh': f'data/{obj}/mesh/vase.obj',
        'frames': f'data/{obj}/frames_from_video',
        'n_frames': nfr}]
p = E/'out/orient/orientation_vase.json'
p.write_text(json.dumps(out, indent=2))
print(f'== orient: wrote {p}\n{np.round(R,4)}')
PY

# ---- 2b. ALIGNMENT GATE, on CPU, before a single GPU-second is spent --------
# Non-negotiable: a wrong pose renders a plausible silhouette and trains for four
# hours into garbage. check_vase_align_cpu.py rasterises our posed mesh through the
# pipeline camera and compares to the video silhouette. Its rasteriser is calibrated
# against nvdiffrast's own render_mask.npy (calib_cpu_raster.py) and it reproduces
# all seven recorded solve IoUs to within 0.002 (validate_align_gate.py) -- an
# uncalibrated version of this check produced a FALSE FAIL and cost two cancelled jobs.
echo "== alignment gate"
$PY2 "$E/check_vase_align_cpu.py" 2>&1 | tee "$E/out/check_vase_align_cpu.log"
grep -q "VERDICT: PASS" "$E/out/check_vase_align_cpu.log" || {
  echo "FATAL: alignment gate failed - NOT submitting. See $E/out/check_vase_align_cpu.log"; exit 1; }
$PY2 - <<'PYG'
import json, re
from pathlib import Path
E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
iou = float(re.search(r'best IoU\s+([0-9.]+)', (E/'out/check_vase_align_cpu.log').read_text()).group(1))
p = E/'out/orient/orientation_vase.json'
d = json.loads(p.read_text()); d[0]['iou_after'] = iou      # measured, not a sentinel
d[0]['verdict'] = 'KNOWN_ROT_GATED'
p.write_text(json.dumps(d, indent=2)); print(f'== gate IoU {iou:.4f} recorded in orient json')
PYG

# ---- 3. targets (GPU, minutes), then training, chained ---------------------
EXP1="OBJ=${OBJ},ORIENT=${ORIENT},SUFFIX="
J1=$(sbatch --parsable --job-name="mkt_${OBJ}" --export=ALL,"$EXP1" "$E/jobs/build_targets_obj.sbatch")
note "$J1" "mkt_${OBJ}" "$EXP1" "jobs/build_targets_obj.sbatch" \
  "TARGETS for ${OBJ} from KNOWN rotation (yaw240/pitch25, byte-exact vs kling png); writes data/${OBJ}/mesh/${OBJ}_render_frame.obj + out/gt_targets_${OBJ}/frames; out=$E/out/mkt_${J1}.log; PASS: log has '[OK] ${OBJ}' and out/gt_targets_${OBJ}/frames has ${NFR} pngs"

MESH=$T2/data/$OBJ/mesh/${OBJ}_render_frame.obj
GTDIR=$E/out/gt_targets_${OBJ}/frames
EXP2="OBJ=${OBJ},NFR=${NFR},MODE=v2_D,MESH=${MESH},GTDIR=${GTDIR}"
J2=$(sbatch --parsable --job-name="m3_${OBJ}_v2_D" --dependency=afterok:$J1 \
      --export=ALL,"$EXP2" "$E/jobs/rung27_mcfm_mode.sbatch")
note "$J2" "m3_${OBJ}_v2_D" "$EXP2" "jobs/rung27_mcfm_mode.sbatch" \
  "rung27 + MCFM temporal_only_w3 (v2_D), ${OBJ}, ${NFR}fr, seed42, 30ep; waits on afterok:${J1}; out=$E/out/r27m3_${J2}.log; PASS: log has '[FINAL] rung27' and runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_*/final_eval.json exists"

echo
echo "chain: $J1 (targets) -> $J2 (rung27+MCFM v2_D)"
echo "watch: $JL running   |   tail -f $E/out/mkt_${J1}.log"
