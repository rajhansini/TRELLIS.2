#!/bin/bash
set -uo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
O=$D/out/R30H_VIDEOS; mkdir -p $O; rm -rf $D/out/_r30hc
$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
D='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT=sorted(glob.glob(f'{D}/out/gt_targets_horse_lava/frames/gt_*.png'))
assert len(GT)>=150, len(GT)
BAR=28
for k in ('pumpkin','teapot'):
    for v in ('side','360','720'):
        src=sorted(glob.glob(f'{D}/out/r30h_{k}_{v}/frames/*.png'))
        assert len(src)==150, f'{k}_{v}: {len(src)}'
        out=f'{D}/out/_r30hc/{k}_{v}'; os.makedirs(out, exist_ok=True)
        for i,p in enumerate(src,1):
            rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
            gt=Image.open(GT[i-1]).convert('RGB').resize((W,H-BAR),Image.LANCZOS)
            pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
            ImageDraw.Draw(pan).text((W//2-104,8),'SOURCE TEXTURE (horse_lava)',fill=(255,255,255))
            c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
            c.save(f'{out}/{i:04d}.png')
        print(f'  {k}_{v}: 150', flush=True)
PYEOF
for k in pumpkin teapot; do for v in side 360 720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $D/out/_r30hc/${k}_${v}/%04d.png \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $O/r30h_${k}_${v}_source_frozen_transfer.mp4 && echo "  encoded ${k}_${v}"
done; done
rm -rf $D/out/_r30hc
echo "R30H_DONE"
