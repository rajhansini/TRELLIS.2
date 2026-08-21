#!/bin/bash
# One video per (object, view): a labelled row of tiles, all frames.
#
# Every rendered frame is [frozen | ours] side by side: 1036x546 = two 518x518
# panels under a 28 px label strip. So the arm tile is the RIGHT half and the
# frozen baseline is the LEFT half.
#
# WHICH FROZEN. Each arm's render carries its OWN frozen half, and they are NOT
# interchangeable -- the MCFM render feeds its frozen arm blended tokens and the
# rung32 render feeds it 3087 concatenated ones. The honest baseline is the frozen
# half from the PLAIN rung27 render, which is the only one that saw vanilla
# single-frame conditioning. Taking it from any other arm would smuggle that arm's
# conditioning into the column labelled "frozen".
#
# GT exists at the TRAINING view only. At yaw 90/180/270 there is no ground truth
# by construction -- that is what makes them unseen -- so those rows carry five
# tiles, not six, and say so rather than padding with something misleading.
set -euo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
OUT=$E/out/panel_rows; mkdir -p "$OUT"
S=518; Y0=28
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
[ -f "$FONT" ] || FONT=$(fc-match -f "%{file}" "DejaVu Sans" 2>/dev/null || echo "")

lbl () { if [ -n "$FONT" ]; then
  echo "drawtext=fontfile=${FONT}:text='$1':x=(w-text_w)/2:y=6:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6"
 else echo "null"; fi }

for OBJ in spot_lava skull_lava hand_rorschach pumpkin_rot; do
  case $OBJ in pumpkin_rot) NF=121; GTD=$E/out/gt_targets_pumpkin_rot_guan/frames;;
               *)           NF=150; GTD=$E/out/gt_targets_${OBJ}/frames;; esac
  for YAW in 0 90 180 270; do
    R27=$E/out/panel_${OBJ}_rung27_y${YAW}/frames
    MCF=$E/out/panel_${OBJ}_mcfm_y${YAW}/frames
    R31=$E/out/panel_${OBJ}_rung31_y${YAW}/frames
    R32=$E/out/panel_${OBJ}_rung32_y${YAW}/frames
    ok=1; for d in "$R27" "$MCF" "$R31" "$R32"; do
      n=$(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      [ "$n" -ge "$NF" ] || { echo "SKIP ${OBJ} y${YAW}: $(basename $(dirname $d)) has $n/$NF"; ok=0; }
    done
    [ $ok -eq 1 ] || continue

    CROP_R="crop=${S}:${S}:${S}:${Y0}"      # right half  = the adapted arm
    CROP_L="crop=${S}:${S}:0:${Y0}"         # left half   = frozen
    O=$OUT/row_${OBJ}_y${YAW}.mp4

    if [ "$YAW" = "0" ] && [ -d "$GTD" ]; then
      ffmpeg -y -loglevel error \
        -framerate 20 -i "$GTD/gt_%04d.png" \
        -framerate 20 -i "$R27/%04d.png" -framerate 20 -i "$R27/%04d.png" \
        -framerate 20 -i "$MCF/%04d.png" -framerate 20 -i "$R31/%04d.png" \
        -framerate 20 -i "$R32/%04d.png" \
        -filter_complex "\
[0:v]scale=${S}:${S},$(lbl 'GT (training view)')[a];\
[1:v]${CROP_L},$(lbl 'frozen')[b];\
[2:v]${CROP_R},$(lbl 'rung27')[c];\
[3:v]${CROP_R},$(lbl 'rung27+MCFM')[d];\
[4:v]${CROP_R},$(lbl 'rung31 dual+tw3')[e];\
[5:v]${CROP_R},$(lbl 'rung32 cw3')[f];\
[a][b][c][d][e][f]hstack=inputs=6,scale=trunc(iw/4)*2:trunc(ih/4)*2[v]" \
        -map "[v]" -frames:v "$NF" -c:v libx264 -pix_fmt yuv420p -crf 20 "$O"
    else
      ffmpeg -y -loglevel error \
        -framerate 20 -i "$R27/%04d.png" -framerate 20 -i "$R27/%04d.png" \
        -framerate 20 -i "$MCF/%04d.png" -framerate 20 -i "$R31/%04d.png" \
        -framerate 20 -i "$R32/%04d.png" \
        -filter_complex "\
[0:v]${CROP_L},$(lbl "frozen  (yaw ${YAW}, unseen)")[b];\
[1:v]${CROP_R},$(lbl 'rung27')[c];\
[2:v]${CROP_R},$(lbl 'rung27+MCFM')[d];\
[3:v]${CROP_R},$(lbl 'rung31 dual+tw3')[e];\
[4:v]${CROP_R},$(lbl 'rung32 cw3')[f];\
[b][c][d][e][f]hstack=inputs=5,scale=trunc(iw/4)*2:trunc(ih/4)*2[v]" \
        -map "[v]" -frames:v "$NF" -c:v libx264 -pix_fmt yuv420p -crf 20 "$O"
    fi
    echo "built $(basename $O)  $(du -h "$O" | cut -f1)"
  done
done
echo "rows in $OUT: $(ls $OUT/*.mp4 2>/dev/null | wc -l)"
