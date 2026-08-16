#!/bin/bash
# Prepend the GT target as a left panel to the 18 rung18/19/25 renders.
# Output: out/ALL_GT_VIDEOS/<tag>_<view>_GT_frozen_adapted.mp4  (GT | frozen | ours)
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/ALL_GT_VIDEOS; mkdir -p $OUT
TAGS="spot_r18 teap_r18 spot_r19 teap_r19 spot_r25 teap_r25"

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT={'spot':f'{E}/out/gt_targets_spot_lava/frames','teap':f'{E}/out/gt_targets_teapot_lava2/frames'}
BAR=28
for t in ('spot_r18','teap_r18','spot_r19','teap_r19','spot_r25','teap_r25'):
    obj=t.split('_')[0]
    gts=sorted(glob.glob(f'{GT[obj]}/*.png'))
    assert len(gts)>=150, f'{obj}: only {len(gts)} GT frames'
    for v in ('side','360','720'):
        src=sorted(glob.glob(f'{E}/out/render_{t}_{v}/frames/*.png'))
        assert len(src)==150, f'{t}_{v}: {len(src)} render frames'
        out=f'{E}/out/_gtc/{t}_{v}'; os.makedirs(out,exist_ok=True)
        for i,p in enumerate(src,1):
            rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
            gt=Image.open(gts[i-1]).convert('RGB').resize((W,H-BAR),Image.LANCZOS)
            pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
            ImageDraw.Draw(pan).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
            c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
            c.save(f'{out}/{i:04d}.png')
        print(f'  composited {t}_{v}: {len(src)} frames', flush=True)
PYEOF

for t in $TAGS; do for v in side 360 720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $E/out/_gtc/${t}_${v}/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/${t}_${v}_GT_frozen_adapted.mp4 && echo "  encoded ${t}_${v}"
done; done
rm -rf $E/out/_gtc
echo "[DONE]"
