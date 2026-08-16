#!/bin/bash
# Adds the GROUND TRUTH panel to the six rung27 renders and encodes them.
# Same 3-panel layout as every video already in out/ALL_GT_VIDEOS:
#     GROUND TRUTH  |  TRELLIS.2 frozen  |  rung27 adapted
# The renderer emits 2 panels (frozen | adapted); the GT panel is composited on
# here from the backprojected targets, so it is the SAME image the loss scored.
# CPU only — no GPU, no model.
#SBATCH --job-name=r27gt
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27gt_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/r27gt_%j.log
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OUT=$E/out/ALL_GT_VIDEOS; mkdir -p "$OUT"
echo "=== $(date) host=$(hostname) ==="

# The dependency already guarantees the renders finished, but a render can exit 0
# with fewer than 150 frames if it was requeued mid-way. Check the count, do not
# assume it.
for t in r27spot r27teap; do for v in side 360 720; do
  n=$(ls $E/out/render_${t}_${v}/frames/*.png 2>/dev/null | wc -l)
  echo "  render_${t}_${v}: $n frames"
  [ "$n" -ge 150 ] || { echo "FATAL: render_${t}_${v} has $n/150 frames"; exit 1; }
done; done

$PY - <<'PYEOF'
import os, glob
from PIL import Image, ImageDraw
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
GT={'r27spot':f'{E}/out/gt_targets_spot_lava/frames',
    'r27teap':f'{E}/out/gt_targets_teapot_lava2/frames'}
BAR=28
for t in ('r27spot','r27teap'):
    for v in ('side','360','720'):
        src=f'{E}/out/render_{t}_{v}/frames'
        files=sorted(glob.glob(src+'/*.png'))
        out=f'{E}/out/_gtc/{t}_{v}'; os.makedirs(out,exist_ok=True)
        for i,p in enumerate(files,1):
            rest=Image.open(p).convert('RGB'); W=rest.width//2; H=rest.height
            gt=Image.open(f'{GT[t]}/gt_{i:04d}.png').convert('RGB').resize((W,H-BAR),Image.LANCZOS)
            pan=Image.new('RGB',(W,H),(26,27,35)); pan.paste(gt,(0,BAR))
            ImageDraw.Draw(pan).text((W//2-52,8),'GROUND TRUTH',fill=(255,255,255))
            c=Image.new('RGB',(W+rest.width,H)); c.paste(pan,(0,0)); c.paste(rest,(W,0))
            c.save(f'{out}/{i:04d}.png')
        print(f'  composited {t}_{v}: {len(files)}', flush=True)
PYEOF

for t in r27spot r27teap; do for v in side 360 720; do
  d=$E/out/_gtc/${t}_${v}
  ffmpeg -nostdin -v error -y -framerate 20 -i $d/%04d.png \
    -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -movflags +faststart \
    $OUT/${t}_${v}_GT_frozen_adapted.mp4
  echo "  encoded ${t}_${v}  $(du -h $OUT/${t}_${v}_GT_frozen_adapted.mp4 | cut -f1)"
done; done

echo "[DONE] $(ls $OUT/*.mp4 | wc -l) videos total in $OUT"
ls -la $OUT/r27*
echo "=== Done $(date) ==="
