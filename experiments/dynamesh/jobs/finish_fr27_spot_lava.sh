#!/bin/bash
# Waits for the four r27v_spot_lava renders, then recomputes that object's
# arm-27 video-space metrics. The stale json must be removed first: both
# submit_ladder_fullrate.sh and fullrate_obj_arm.sbatch are idempotent on the
# output file and would otherwise skip the very cell we are trying to fix.
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
cd "$E"
for i in $(seq 1 180); do
  D=0
  for V in train diagA diagB diagC; do
    C=0; [ -d "out/view_spot_lava_27_$V/frames" ] && \
      C=$(find "out/view_spot_lava_27_$V/frames" -maxdepth 1 -name '*.png' | wc -l)
    [ "$C" -ge 150 ] && D=$((D+1))
  done
  [ "$D" -eq 4 ] && break
  Q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c '^r27v_spot_lava' || true)
  [ "$Q" -eq 0 ] && [ "$i" -gt 3 ] && { echo "RENDERS STOPPED with only $D/4 complete"; exit 1; }
  sleep 30
done
echo "renders complete, recomputing metrics"
rm -f out/FULLRATE_R27/spot_lava.json
ONLY=spot_lava ARMS=27 bash jobs/submit_ladder_fullrate.sh
for i in $(seq 1 60); do
  [ -s out/FULLRATE_R27/spot_lava.json ] && break; sleep 20
done
python3 -c "
import json
for d,l in [('FULLRATE_R19','r19'),('FULLRATE_R27','r27'),('FULLRATE_CG','27m'),('FULLRATE_W11CG','27g')]:
    j=json.load(open('out/%s/spot_lava.json'%d))['spot_lava']['full']['train|ours']
    print('%-4s psnr=%.2f flicker=%.5f'%(l,j['psnr'],j['flicker']))
"
