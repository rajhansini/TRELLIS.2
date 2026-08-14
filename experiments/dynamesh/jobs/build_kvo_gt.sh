#!/bin/bash
# Adds the GROUND TRUTH panel to all three views of the kvo run (23.37 dB).
# Its 360/720 frames already exist; only the side view is being rendered.
# Targets are the 518^2 set this run trained on (out/gt_targets), NOT the 960
# ones — this run predates render_res and used the default.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/ALL_GT_VIDEOS
until [ "$(ls $E/out/render_kvo_side/frames/*.png 2>/dev/null | wc -l)" -ge 150 ]; do sleep 30; done
echo "[READY] side view rendered"

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT=f'{E}/out/gt_targets/frames'          # 518^2 targets this run trained on
BAR=28
SRC=[('side', f'{E}/out/render_kvo_side/frames'),
     ('360',  f'{E}/out/rung18_1x360/frames'),
     ('720',  f'{E}/out/rung18_2x360/frames')]
for v, src in SRC:
    files=sorted(glob.glob(src+'/*.png'))
    if len(files) < 150:
        print(f'  SKIP {v}: only {len(files)} frames'); continue
    out=f'{E}/out/_gtc/kvo_{v}'; os.makedirs(out, exist_ok=True)
    for i,p in enumerate(files,1):
        rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
        gt=Image.open(f'{GT}/gt_{i:04d}.png').convert('RGB').resize((W,H-BAR),Image.LANCZOS)
        pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
        ImageDraw.Draw(pan).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
        c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
        c.save(f'{out}/{i:04d}.png')
    print(f'  composited kvo_{v}: {len(files)} frames -> {c.size}', flush=True)
PYEOF

for v in side 360 720; do
  d=$E/out/_gtc/kvo_$v
  [ -d "$d" ] || continue
  ffmpeg -nostdin -v error -y -framerate 20 -i $d/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/kvo_${v}_GT_frozen_adapted.mp4
  echo "  encoded kvo_${v}"
done
echo "[DONE]"; ls -la $OUT/kvo_*.mp4
