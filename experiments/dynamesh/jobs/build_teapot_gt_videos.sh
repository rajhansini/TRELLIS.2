#!/bin/bash
# Waits for the 6 render jobs, composites the GT target as a left panel onto
# every one, encodes at crf16, and drops all six into out/TEAPOT_GT_VIDEOS/.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/TEAPOT_GT_VIDEOS
mkdir -p $OUT

for t in teapot960_side teapot960_360 teapot960_720 lava2_side lava2_360 lava2_720; do
  until [ "$(ls $E/out/render_${t}/frames/*.png 2>/dev/null | wc -l)" -ge 150 ]; do sleep 30; done
  echo "[READY] render_${t}"
done
echo "[ALL RENDERS DONE] $(date)"

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
S=f'{E}/out/_gtcomp'; BAR=28
GT={'teapot960':f'{E}/out/gt_targets_teapot_960/frames',
    'lava2':    f'{E}/out/gt_targets_teapot_lava2/frames'}
for t in ('teapot960_side','teapot960_360','teapot960_720',
          'lava2_side','lava2_360','lava2_720'):
    obj=t.rsplit('_',1)[0]
    src=sorted(glob.glob(f'{E}/out/render_{t}/frames/*.png'))
    out=f'{S}/{t}'; os.makedirs(out, exist_ok=True)
    for i,p in enumerate(src,1):
        rest=Image.open(p).convert('RGB')
        W=rest.width//2; H=rest.height
        gt=Image.open(f'{GT[obj]}/gt_{i:04d}.png').convert('RGB').resize((W,H-BAR), Image.LANCZOS)
        panel=Image.new('RGB',(W,H),(26,27,35)); panel.paste(gt,(0,BAR))
        ImageDraw.Draw(panel).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
        comp=Image.new('RGB',(W+rest.width,H)); comp.paste(panel,(0,0)); comp.paste(rest,(W,0))
        comp.save(f'{out}/{i:04d}.png')
    print(f'  composited {t}: {len(src)} frames -> {comp.size}', flush=True)
PYEOF

for t in teapot960_side teapot960_360 teapot960_720 lava2_side lava2_360 lava2_720; do
  ffmpeg -nostdin -v error -y -framerate 20 -i $E/out/_gtcomp/$t/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/${t}_GT_frozen_adapted.mp4
  printf "  %-22s %6.2f MB\n" $t $(echo "scale=2;$(stat -c%s $OUT/${t}_GT_frozen_adapted.mp4)/1048576"|bc)
done
echo "[DONE] $OUT"; ls -la $OUT
