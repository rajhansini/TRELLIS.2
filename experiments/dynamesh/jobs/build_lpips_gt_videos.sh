#!/bin/bash
# Waits for the 12 LPIPS renders, adds the GT panel, encodes into ALL_GT_VIDEOS.
# Also folds in spot_lava rung20 L2 (fe2edd55), whose renders exist as
# render_lava_* from an earlier pass but were never given a GT panel.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/ALL_GT_VIDEOS; mkdir -p $OUT
NEW="spot_r17lp spot_r20lp teap_r17lp teap_r20lp"
for t in $NEW; do for v in side 360 720; do
  until [ "$(ls $E/out/render_${t}_${v}/frames/*.png 2>/dev/null | wc -l)" -ge 150 ]; do sleep 60; done
  echo "[READY] ${t}_${v}"
done; done
echo "[RENDERS DONE] $(date)"

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT={'spot':f'{E}/out/gt_targets_spot_lava/frames','teap':f'{E}/out/gt_targets_teapot_lava2/frames'}
BAR=28
jobs=[(t,v,f'{E}/out/render_{t}_{v}/frames')
      for t in ('spot_r17lp','spot_r20lp','teap_r17lp','teap_r20lp')
      for v in ('side','360','720')]
# spot_lava rung20 L2: renders already exist under the old naming
for v,src in (('side','render_lava_side'),('360','render_lava_360'),('720','render_lava_720')):
    p=f'{E}/out/{src}/frames'
    if len(glob.glob(p+'/*.png'))>=150: jobs.append(('spot_r20l2',v,p))
for t,v,src in jobs:
    obj='spot' if t.startswith('spot') else 'teap'
    files=sorted(glob.glob(src+'/*.png'))
    out=f'{E}/out/_gtc/{t}_{v}'; os.makedirs(out,exist_ok=True)
    for i,p in enumerate(files,1):
        rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
        gt=Image.open(f'{GT[obj]}/gt_{i:04d}.png').convert('RGB').resize((W,H-BAR),Image.LANCZOS)
        pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
        ImageDraw.Draw(pan).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
        c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
        c.save(f'{out}/{i:04d}.png')
    print(f'  composited {t}_{v}: {len(files)}', flush=True)
PYEOF

for d in $E/out/_gtc/*; do
  b=$(basename $d)
  [ -f "$OUT/${b}_GT_frozen_adapted.mp4" ] && continue
  ffmpeg -nostdin -v error -y -framerate 20 -i $d/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/${b}_GT_frozen_adapted.mp4
done
echo "[DONE] $(ls $OUT/*.mp4 | wc -l) videos"; ls $OUT
