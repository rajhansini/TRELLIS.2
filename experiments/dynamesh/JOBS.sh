#!/bin/bash
# Standing registry of every experiment. Run this any time instead of asking:
#   bash experiments/dynamesh/JOBS.sh
# Reads SLURM and the run directories directly, so it is never stale.
D=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
echo "==================== RUNNING / QUEUED ===================="
squeue -u rajhansini -o "%.10i %.16j %.9T %.7M %.9L %.6R" 2>/dev/null
echo ""
echo "==================== FINISHED RUNS (30 epochs) ===================="
python3 - <<'PY'
import json, glob, os
R='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/runs'
rows=[]
for d in glob.glob(R+'/*'):
    fe,cf=d+'/final_eval.json',d+'/config.json'
    if not (os.path.exists(fe) and os.path.exists(cf)): continue
    try: c=json.load(open(cf)); c=c.get('cfg',c); f=json.load(open(fe))
    except Exception: continue
    if c.get('epochs')!=30: continue
    gt=c.get('gt_dir') or ''
    obj=('spot_lava' if 'spot_lava' in gt else 'spot_star' if 'spot_star' in gt
         else 'teapot_lava2' if 'teapot_lava2' in gt else 'teapot(orig)')
    loss=c.get('recon','l2')+('+lp' if c.get('lpips') else '')
    rows.append((f['final']['psnr_mean'], obj, c.get('rung'), c.get('targets'),
                 'w' if c.get('conf_weight') else '-', loss,
                 c.get('render_res'), f['frozen']['psnr_mean'],
                 f['final']['ssim_mean'], os.path.basename(d)))
rows.sort(key=lambda r:-r[0])
print(f'{"PSNR":>7}{"frozen":>8}{"ssim":>7}  {"object":<13}{"rung":>5}{"targets":>8}{"w":>3}{"loss":>8}{"res":>5}  dir')
for r in rows:
    print(f'{r[0]:>7.2f}{r[7]:>8.2f}{r[8]:>7.3f}  {r[1]:<13}{str(r[2]):>5}{str(r[3]):>8}{r[4]:>3}{r[5]:>8}{str(r[6]):>5}  {r[9]}')
print(f'\n  {len(rows)} completed 30-epoch runs')
PY
echo ""
echo "==================== VIDEOS ===================="
n=$(ls $D/out/ALL_GT_VIDEOS/*.mp4 2>/dev/null | wc -l)
echo "  $n in out/ALL_GT_VIDEOS/   (GT | frozen | ours, 3 views per run)"
echo "  scp -r rajhansini@fe02.ai.cs.uchicago.edu:$D/out/ALL_GT_VIDEOS ~/Downloads/"
echo ""
echo "==================== SAM EXPERIMENT (isolated) ===================="
echo "  env      /net/projects/ranalab/rajhansini/conda_envs/sam_env  (sam2 1.1.0)"
echo "  folder   $D/sam_experiment/"
ls $D/sam_experiment/out/*.log 2>/dev/null | tail -2 | sed 's|^|  log      |'
