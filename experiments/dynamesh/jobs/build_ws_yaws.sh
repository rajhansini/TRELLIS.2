#!/bin/bash
# Wait for the 12 fixed-view renders, then composite 4 yaw strips and encode.
set -uo pipefail
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
JOBS=$(seq -s, 2190510 2190521)
SKIPWAIT=${SKIPWAIT:-0}
if [ "$SKIPWAIT" = 0 ]; then
for i in $(seq 1 480); do
  st=$(sacct -j $JOBS -X --format=State -n 2>/dev/null | tr -d ' ')
  n_done=$(echo "$st" | grep -c COMPLETED)
  n_bad=$(echo "$st" | grep -cE 'FAILED|CANCELLED|TIMEOUT')
  [ $((n_done+n_bad)) -ge 12 ] && break
  sleep 30
done
fi
echo "[WAIT] done=${n_done:-skipped} bad=${n_bad:-0}"
[ "${n_bad:-0}" -gt 0 ] && { echo "[ABORT] $n_bad failed"; exit 1; }

$PY - <<'PYEOF'
import glob, os
import numpy as np
from PIL import Image, ImageDraw
D='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
BAR=30
LAB=['FROZEN TRELLIS.2','baseline  24.574 dB','+ mask crop  24.998 dB','+ warm start  25.996 dB']
TITLE={'000':'yaw 0deg  -  TRAINING VIEW','120':'yaw 120deg  -  UNSEEN',
       '180':'yaw 180deg  -  UNSEEN','240':'yaw 240deg  -  UNSEEN'}
for y in ('000','120','180','240'):
    fa=sorted(glob.glob(f'{D}/out/ws_base_yaw{y}/frames/*.png'))
    fb=sorted(glob.glob(f'{D}/out/ws_cold_yaw{y}/frames/*.png'))
    fc=sorted(glob.glob(f'{D}/out/ws_warm_yaw{y}/frames/*.png'))
    n=min(len(fa),len(fb),len(fc))
    assert n>=121, (y,len(fa),len(fb),len(fc))
    out=f'{D}/out/_wsy/{y}'; os.makedirs(out,exist_ok=True)
    for i in range(n):
        A=np.asarray(Image.open(fa[i]).convert('RGB'))
        B=np.asarray(Image.open(fb[i]).convert('RGB'))
        C=np.asarray(Image.open(fc[i]).convert('RGB'))
        H,W,_=A.shape; h=W//2
        strip=np.concatenate([A[:,:h],A[:,h:],B[:,h:],C[:,h:]],axis=1)
        canv=Image.new('RGB',(strip.shape[1],H+BAR),(22,23,30))
        canv.paste(Image.fromarray(strip),(0,BAR))
        d=ImageDraw.Draw(canv)
        for j,t in enumerate(LAB): d.text((j*h+8,9),t,fill=(240,240,240))
        d.text((strip.shape[1]-230,9), f'{TITLE[y]}   frame {i+1}/{n}', fill=(175,175,185))
        canv.save(f'{out}/{i+1:04d}.png')
    print(f'  yaw{y}: {n} frames {strip.shape[1]}x{H+BAR}', flush=True)
PYEOF

O=$D/out/WS_YAWS; mkdir -p $O
for y in 000 120 180 240; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $D/out/_wsy/$y/%04d.png \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $O/penguin_yaw${y}.mp4 && echo "  encoded yaw$y"
done
W=$D/out/_wsyweb; mkdir -p $W; rm -f $W/*.mp4
for y in 000 120 180 240; do
  ffmpeg -nostdin -v error -y -i $O/penguin_yaw${y}.mp4 -vf "scale=1800:-2" \
    -c:v libx264 -crf 27 -preset slow -pix_fmt yuv420p -movflags +faststart -an $W/$y.mp4 &
done; wait
rm -rf $D/out/_wsy
du -sh $W | sed 's/^/  web total /'
echo "WS_YAWS_DONE"
