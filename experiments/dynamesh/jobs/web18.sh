#!/bin/bash
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
W=$D/out/_web18; rm -rf $W; mkdir -p $W
for k in spot_r18 teap_r18 spot_r19 teap_r19 spot_r25 teap_r25; do
  for v in side 360 720; do
    ffmpeg -nostdin -v error -y -i $D/out/ALL_GT_VIDEOS/${k}_${v}_GT_frozen_adapted.mp4 \
      -vf "scale=1728:-2" -c:v libx264 -crf 27 -preset slow -pix_fmt yuv420p \
      -movflags +faststart -an $W/${k}_${v}.mp4 && echo "ok ${k}_${v}"
  done
done
echo "WEB18_DONE"
