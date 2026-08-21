#!/bin/bash
# One video per object: the KL beta diagonal as a labelled row, all frames.
#
# WHAT THE ROW SHOWS. Cell 1 is the in-grid control (w_kl = w_kl_self = 0), then
# four steps down the diagonal of the 4x4 grid with BOTH betas rising together.
# PSNR falls monotonically along it on every one of the nine objects swept, so the
# row is the visual form of that table: if the claim is real the texture should
# visibly wash out left to right.
#
# The tiles come from the SAME renders as the temporal panels -- fixed camera at the
# training view, 518 px, all frames -- so the two sections are directly comparable
# and neither is doing something the other is not.
set -euo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
OUT=$E/out/panel_rows; mkdir -p "$OUT"
S=518; Y0=28
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
[ -f "$FONT" ] || FONT=$(fc-match -f "%{file}" "DejaVu Sans" 2>/dev/null || echo "")
lbl () { if [ -n "$FONT" ]; then
  echo "drawtext=fontfile=${FONT}:text='$1':x=(w-text_w)/2:y=6:fontsize=21:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6"
 else echo "null"; fi }

for OBJ in spot_lava skull_lava; do
  NF=150; ins=(); filt=""; i=0
  while IFS=$'\t' read -r O C BC BS J TAG PS N; do
    [ "$O" = "$OBJ" ] || continue
    d=$E/out/$TAG/frames
    n=$(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
    [ "$n" -ge "$NF" ] || { echo "SKIP $OBJ: $TAG has $n/$NF"; continue 2; }
    ins+=(-framerate 20 -i "$d/%04d.png")
    if [ "$C" = "ctrl" ]; then T="control  β=0   ${PS}dB"
    else T="βc ${BC}  βs ${BS}   ${PS}dB"; fi
    filt="${filt}[${i}:v]crop=${S}:${S}:${S}:${Y0},$(lbl "$T")[t${i}];"
    i=$((i+1))
  done < $E/out/kl_panel_jobs.tsv
  [ "$i" -eq 5 ] || { echo "SKIP $OBJ: only $i/5 tiles"; continue; }
  refs=""; for k in $(seq 0 4); do refs="${refs}[t${k}]"; done
  ffmpeg -y -loglevel error "${ins[@]}" \
    -filter_complex "${filt}${refs}hstack=inputs=5,scale=trunc(iw/4)*2:trunc(ih/4)*2[v]" \
    -map "[v]" -frames:v "$NF" -c:v libx264 -pix_fmt yuv420p -crf 20 \
    "$OUT/klrow_${OBJ}.mp4"
  echo "built klrow_${OBJ}.mp4  $(du -h "$OUT/klrow_${OBJ}.mp4" | cut -f1)"
done
