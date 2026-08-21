#!/bin/bash
# Per object, one video per beta_cross level: 4 tiles across the beta_self axis.
# Four of these stacked IS the 4x4 grid, which is the form the sweep was run in --
# a diagonal strip showed 5 of 16 cells and misrepresented it as the grid.
set -euo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
OUT=$E/out/panel_rows; mkdir -p "$OUT"
S=518; Y0=28; NF=150
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
[ -f "$FONT" ] || FONT=$(fc-match -f "%{file}" "DejaVu Sans" 2>/dev/null || echo "")
lbl () { if [ -n "$FONT" ]; then
  echo "drawtext=fontfile=${FONT}:text='$1':x=(w-text_w)/2:y=6:fontsize=21:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6"
 else echo "null"; fi }
for OBJ in spot_lava skull_lava; do
  for I in 1 2 3 4; do
    ins=(); filt=""; k=0
    while IFS=$'\t' read -r O i j BC BS JOB TAG PS N; do
      [ "$O" = "$OBJ" ] && [ "$i" = "$I" ] || continue
      d=$E/out/$TAG/frames
      n=$(find "$d" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
      [ "$n" -ge "$NF" ] || { echo "SKIP ${OBJ} c${I}: $TAG $n/$NF"; continue 3; }
      ins+=(-framerate 20 -i "$d/%04d.png")
      filt="${filt}[${k}:v]crop=${S}:${S}:${S}:${Y0},$(lbl "βs ${BS}   ${PS}dB")[t${k}];"
      k=$((k+1))
    done < $E/out/kl_grid_jobs.tsv
    [ "$k" -eq 4 ] || { echo "SKIP ${OBJ} c${I}: $k/4"; continue; }
    refs=""; for m in 0 1 2 3; do refs="${refs}[t${m}]"; done
    ffmpeg -y -loglevel error "${ins[@]}" \
      -filter_complex "${filt}${refs}hstack=inputs=4,scale=trunc(iw/5)*2:trunc(ih/5)*2[v]" \
      -map "[v]" -frames:v "$NF" -c:v libx264 -pix_fmt yuv420p -crf 24 \
      "$OUT/klgrid_${OBJ}_c${I}.mp4"
    echo "built klgrid_${OBJ}_c${I}.mp4 $(du -h "$OUT/klgrid_${OBJ}_c${I}.mp4"|cut -f1)"
  done
done
