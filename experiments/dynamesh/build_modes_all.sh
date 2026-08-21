#!/bin/bash
# Build every (object, view) 3-mode panel whose renders are complete, then encode
# for web. Idempotent: skips cells already built, so it can be re-run as renders land.
cd /net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
FF=/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin/ffmpeg
mkdir -p out/MODES out/MODES_WEB
declare -A LAB=( [train]="training view" [diagA]="yaw 45° · elev +25°" \
                 [diagB]="yaw 135° · elev −20°" [diagC]="yaw 225° · elev +30°" )
built=0
for o in spot_lava skull_lava hand_rorschach pumpkin_rot; do
  N=150; [ "$o" = pumpkin_rot ] && N=121
  for v in train diagA diagB diagC; do
    [ -s "out/MODES/${o}_${v}_GT_3modes.mp4" ] && continue
    ok=1
    for c in v2_D st_D v3_D; do
      f=$(find "out/m3_${o}_${c}_${v}/frames" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      [ "$f" -ge "$N" ] || ok=0
    done
    [ "$ok" -eq 1 ] || continue
    S=""; [ "$v" = train ] && S="--supervised"
    $PY build_mode_panel.py --obj "$o" --view "$v" --n "$N" --label "${LAB[$v]}" $S && built=$((built+1))
  done
done
for f in out/MODES/*.mp4; do
  [ -s "$f" ] || continue
  w=out/MODES_WEB/$(basename "$f")
  [ -f "$w" ] || $FF -nostdin -v error -y -i "$f" -vf scale=960:-2 -c:v libx264 -crf 34 \
      -preset slow -pix_fmt yuv420p -movflags +faststart "$w"
done
echo "built $built new; total panels $(ls out/MODES/*.mp4 2>/dev/null|wc -l)/16, web $(ls out/MODES_WEB/*.mp4 2>/dev/null|wc -l)"
