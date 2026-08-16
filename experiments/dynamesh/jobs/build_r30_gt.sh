#!/bin/bash
# GT panel for the rung30 transfer = spot_lava's video, i.e. the SOURCE texture.
# There is no horse ground truth: the horse was never trained on. The left panel
# shows what texture is being carried, the right two show where it landed.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/R30_VIDEOS; mkdir -p $OUT
$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT=sorted(glob.glob(f'{E}/out/gt_targets_spot_lava/frames/*.png'))
assert len(GT)>=150, f'only {len(GT)} source frames'
BAR=28
for v in ('side','360','720'):
    src=sorted(glob.glob(f'{E}/out/r30_horse_guan_{v}/frames/*.png'))
    assert len(src)==150, f'{v}: {len(src)} frames'
    out=f'{E}/out/_r30c/{v}'; os.makedirs(out,exist_ok=True)
    for i,p in enumerate(src,1):
        rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
        gt=Image.open(GT[i-1]).convert('RGB').resize((W,H-BAR),Image.LANCZOS)
        pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
        ImageDraw.Draw(pan).text((W//2-96,8),'SOURCE TEXTURE (spot_lava)',fill=(255,255,255))
        c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
        c.save(f'{out}/{i:04d}.png')
    print(f'  composited {v}: {len(src)} frames', flush=True)
PYEOF
for v in side 360 720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $E/out/_r30c/${v}/%04d.png \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/r30_horse_guan_${v}_source_frozen_transfer.mp4 && echo "  encoded $v"
done
rm -rf $E/out/_r30c
echo "R30_GT_DONE"
