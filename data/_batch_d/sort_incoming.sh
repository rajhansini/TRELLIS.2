#!/bin/bash
# Move uploaded Kling mp4s from incoming/ into the per-asset layout every config.json
# expects: data/<asset>/video/<asset>.mp4
set -euo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2/data
n=0
for f in "$D"/_batch_d/incoming/*.mp4; do
  [ -e "$f" ] || { echo "nothing in incoming/"; exit 0; }
  a=$(basename "$f" .mp4)
  if [ ! -d "$D/$a" ]; then echo "SKIP $a — no asset dir (name mismatch?)"; continue; fi
  mv "$f" "$D/$a/video/$a.mp4"
  echo "$a -> $D/$a/video/$a.mp4"
  n=$((n+1))
done
echo "moved $n"
