#!/bin/bash
# Waits for the 18 renders, prepends the GT target as a left panel, encodes,
# and drops everything into out/ALL_GT_VIDEOS/ as GT | frozen | adapted.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/ALL_GT_VIDEOS; mkdir -p $OUT
TAGS="spot_r17l2 spot_r17l1 spot_r20l1 teap_r17l2 teap_r17l1 teap_r20l1"
for t in $TAGS; do for v in side 360 720; do
  until [ "$(ls $E/out/render_${t}_${v}/frames/*.png 2>/dev/null | wc -l)" -ge 150 ]; do sleep 60; done
  echo "[READY] ${t}_${v}"
done; done
echo "[ALL RENDERS DONE] $(date)"

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT={'spot':f'{E}/out/gt_targets_spot_lava/frames','teap':f'{E}/out/gt_targets_teapot_lava2/frames'}
BAR=28
for t in ('spot_r17l2','spot_r17l1','spot_r20l1','teap_r17l2','teap_r17l1','teap_r20l1'):
    obj=t.split('_')[0]
    for v in ('side','360','720'):
        src=sorted(glob.glob(f'{E}/out/render_{t}_{v}/frames/*.png'))
        out=f'{E}/out/_gtc/{t}_{v}'; os.makedirs(out,exist_ok=True)
        for i,p in enumerate(src,1):
            rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
            gt=Image.open(f'{GT[obj]}/gt_{i:04d}.png').convert('RGB').resize((W,H-BAR),Image.LANCZOS)
            pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
            ImageDraw.Draw(pan).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
            c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
            c.save(f'{out}/{i:04d}.png')
        print(f'  composited {t}_{v}: {len(src)} frames', flush=True)
PYEOF

for t in $TAGS; do for v in side 360 720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $E/out/_gtc/${t}_${v}/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/${t}_${v}_GT_frozen_adapted.mp4
done; done
cp -n $E/out/TEAPOT_GT_VIDEOS/*.mp4 $OUT/ 2>/dev/null || true
echo "[DONE]"; ls -la $OUT
