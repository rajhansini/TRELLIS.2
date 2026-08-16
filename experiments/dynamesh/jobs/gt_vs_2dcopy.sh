#!/bin/bash
set -uo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
T=/net/projects/ranalab/rajhansini/TRELLIS.2/data
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
O=$D/out/GT_VS_2DCOPY; mkdir -p $O; rm -rf $O/_f_*
$PY - <<'PYEOF'
import glob, os
from PIL import Image, ImageDraw
D='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
T='/net/projects/ranalab/rajhansini/TRELLIS.2/data'
O=f'{D}/out/GT_VS_2DCOPY'; BAR=30
for obj in ('pumpkin_lava','horse_lava'):
    gt=sorted(glob.glob(f'{T}/{obj}/frames_from_video/frame_*.png'))
    cp=sorted(glob.glob(f'{D}/out/gt_targets_{obj}/frames/gt_*.png'))
    assert len(gt)==len(cp)==150, f'{obj}: {len(gt)} vs {len(cp)}'
    wd=f'{O}/_f_{obj}'; os.makedirs(wd, exist_ok=True)
    for i,(a,b) in enumerate(zip(gt,cp),1):
        A=Image.open(a).convert('RGB'); B=Image.open(b).convert('RGB').resize(A.size, Image.LANCZOS)
        W,H=A.size
        c=Image.new('RGB',(W*2,H+BAR),(24,25,32)); c.paste(A,(0,BAR)); c.paste(B,(W,BAR))
        d=ImageDraw.Draw(c)
        d.text((W//2-70,9), f'GT VIDEO FRAME  {i}/150', fill=(255,255,255))
        d.text((W+W//2-70,9), '2D COPY (training target)', fill=(255,255,255))
        c.save(f'{wd}/{i:04d}.png')
    print(f'  {obj}: 150 pairs', flush=True)
PYEOF
for obj in pumpkin_lava horse_lava; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $O/_f_$obj/%04d.png \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $O/${obj}_gt_vs_2dcopy.mp4 && echo "  encoded $obj"
  # stills at 4 frames
  $PY -c "
from PIL import Image
import numpy as np
im=[np.asarray(Image.open('$O/_f_$obj/%04d.png'%i)) for i in (1,50,100,150)]
Image.fromarray(np.concatenate(im,axis=0)).resize((im[0].shape[1]//2, sum(x.shape[0] for x in im)//2), Image.LANCZOS).save('$O/${obj}_stills.png')"
  rm -rf $O/_f_$obj
done
echo "GT2D_DONE"
