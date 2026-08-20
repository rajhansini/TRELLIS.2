#!/bin/bash
set -uo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
O=$D/out/R27_VS_R31; mkdir -p $O; rm -rf $D/out/_cmp
$PY - <<'PYEOF'
import glob, os
import numpy as np
from PIL import Image, ImageDraw
D='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
BAR=30
SRC={'side':('render_r27g_penguin_circuits_side','r31_peng_cold_side','r31_peng_warm_side'),
     '360' :('render_r27g_penguin_circuits_360','r31_peng_cold_360','r31_peng_warm_360'),
     '720' :('render_r27g_penguin_circuits_720','r31_peng_cold_720','r31_peng_warm_720')}
LAB=['FROZEN TRELLIS.2','rung27  24.574 dB','rung31 COLD  24.998 dB','rung31 WARM  25.996 dB']
for view,(a,b,c) in SRC.items():
    fa=sorted(glob.glob(f'{D}/out/{a}/frames/*.png'))
    fb=sorted(glob.glob(f'{D}/out/{b}/frames/*.png'))
    fc=sorted(glob.glob(f'{D}/out/{c}/frames/*.png'))
    n=min(len(fa),len(fb),len(fc)); assert n>=121,(view,n)
    out=f'{D}/out/_cmp/{view}'; os.makedirs(out,exist_ok=True)
    for i in range(n):
        A=np.asarray(Image.open(fa[i]).convert('RGB'))
        B=np.asarray(Image.open(fb[i]).convert('RGB'))
        C=np.asarray(Image.open(fc[i]).convert('RGB'))
        H,W,_=A.shape; h=W//2
        panels=[A[:,:h], A[:,h:], B[:,h:], C[:,h:]]     # frozen once, then 3 adapted
        strip=np.concatenate(panels,axis=1)
        canv=Image.new('RGB',(strip.shape[1],H+BAR),(22,23,30))
        canv.paste(Image.fromarray(strip),(0,BAR))
        d=ImageDraw.Draw(canv)
        for j,t in enumerate(LAB):
            d.text((j*h+8,9), f'{t}', fill=(240,240,240))
        d.text((strip.shape[1]-150,9), f'frame {i+1}/{n}', fill=(170,170,180))
        canv.save(f'{out}/{i+1:04d}.png')
    print(f'  {view}: {n} frames  {strip.shape[1]}x{H+BAR}', flush=True)
PYEOF
for v in side 360 720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $D/out/_cmp/$v/%04d.png \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $O/penguin_${v}_r27_vs_r31.mp4 && echo "  encoded $v"
done
rm -rf $D/out/_cmp
echo "CMP_DONE"
