"""
render_ceramic_grid.py — 720 degree turntables for all 16 KL cells, then GT panels.

STAGE 2 of the overnight chain. Waits for every cell to have a [FINAL], submits a
720 render per cell, waits for those, then composites the ground-truth panel on the
left and encodes a web-sized copy for the artifact.

720 MEANS --turns 2 WITH --sweep both: the camera makes two full revolutions while
the frame advances 1..150, so every surface point is seen TWICE at different points
in the texture evolution. That is the view that separates a texture living on the
surface from one painted on at a fixed angle, which is the whole claim.

MCFM. These checkpoints trained on v2_D-blended conditioning. render_rung27_orbit.py
now reads mcfm from the run's own config.json and blends before rendering; before
that patch it fed vanilla tokens and the video showed something never trained.
Verified on job 2183112: GATE-blend max|blended-vanilla| = 0.805.

SIZE. The artifact is already 3.6 MB against a 16 MB cap, so 16 more videos have to
fit in ~12 MB. Each is encoded twice: a full-quality copy kept on disk, and a
web copy targeted at ~600 KB for embedding. The web copy is what goes in the page;
the full one stays for anything that needs it later.
"""
import json, re, subprocess, sys, time
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
OUT, JOBS, RUNS = E / 'out', E / 'jobs', E / 'runs'
OBJ = 'teapot_ceramic_crack_correct'
GT = OUT / f'gt_targets_{OBJ}/frames'
WEB = OUT / 'R30KL_WEB'
FULL = OUT / 'R30KL_FULL'
RC, RS = 5.426, 1.665
FRACS = [15, 25, 50, 75]
NFR = 150
LOG = OUT / 'render_ceramic_grid.log'
DEADLINE = time.time() + 6 * 3600


def log(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def beta(f, r):
    return float(f'{f/100/(1-f/100)*r:.6g}')


# cell -> run dir, matched on the (w_kl, w_kl_self) actually recorded in config.json
cells = {}
for d in RUNS.glob('rung30_l1_lp_mcfmv2_D_*'):
    cf = d / 'config.json'
    if not cf.exists():
        continue
    try:
        c = json.loads(cf.read_text())
    except Exception:
        continue
    if OBJ not in str(c.get('gt_dir', '')):
        continue
    for fc in FRACS:
        for fs in FRACS:
            if (abs(c.get('w_kl', -1) - beta(fc, RC)) < 1e-9
                    and abs(c.get('w_kl_self', -1) - beta(fs, RS)) < 1e-9):
                cells[(fc, fs)] = {'run': d, 'tag': f'r30kl_c{fc}_s{fs}'}
log(f'mapped {len(cells)}/16 cells to run directories')
if len(cells) != 16:
    log('FATAL: could not map all 16 cells; refusing to render a partial grid')
    sys.exit(1)

FINAL = re.compile(r'\[FINAL\] rung\d+  PSNR ([0-9.]+)  SSIM ([0-9.]+)')
HDR = re.compile(r'--w-kl [0-9.eE+-]+ --w-kl-self [0-9.eE+-]+')

# ── wait for every cell to have a final ─────────────────────────────────────
while time.time() < DEADLINE:
    got = {}
    for f in OUT.glob(f'r30g_{OBJ}_*.log'):
        t = f.read_text(errors='replace')
        h, m = HDR.search(t), FINAL.findall(t)
        if h and m:
            got[h.group(0)] = (float(m[-1][0]), float(m[-1][1]))
    for (fc, fs), c in cells.items():
        k = f'--w-kl {beta(fc, RC):g} --w-kl-self {beta(fs, RS):g}'
        if k in got:
            c['psnr'], c['ssim'] = got[k]
    n = sum(1 for c in cells.values() if 'psnr' in c)
    if n == 16:
        log('all 16 cells have finals — starting renders')
        break
    log(f'  waiting: {n}/16 finals')
    time.sleep(180)

# ── submit one 720 render per cell ──────────────────────────────────────────
for (fc, fs), c in cells.items():
    if (OUT / c['tag'] / 'frames').exists() and \
       len(list((OUT / c['tag'] / 'frames').glob('*.png'))) >= NFR:
        log(f'  {c["tag"]}: frames already on disk, skipping')
        c['job'] = None
        continue
    j = sh(f'RUN={c["run"]} TAG={c["tag"]} NFR={NFR} TURNS=2 '
           f'sbatch --parsable {JOBS}/orbit27.sbatch')
    c['job'] = j
    log(f'  render {c["tag"]} -> job {j}')
    subprocess.run(['bash', '/net/projects/ranalab/rajhansini/joblog.sh', 'add', j,
                    f'r30klrender_{c["tag"]}',
                    f'720 turntable (turns=2, sweep=both, {NFR} frames) for KL cell '
                    f'c{fc}_s{fs} on {OBJ}, MCFM-aware renderer | PASS: {NFR} frames '
                    f'+ mp4 in out/{c["tag"]}'], capture_output=True)

# ── wait for renders ────────────────────────────────────────────────────────
while time.time() < DEADLINE:
    n = sum(1 for c in cells.values()
            if len(list((OUT / c['tag'] / 'frames').glob('*.png'))) >= NFR)
    if n == 16:
        log('all 16 renders have frames')
        break
    log(f'  rendering: {n}/16 complete')
    time.sleep(180)

# ── composite GT panel + encode ─────────────────────────────────────────────
WEB.mkdir(exist_ok=True)
FULL.mkdir(exist_ok=True)
from PIL import Image, ImageDraw            # noqa: E402

for (fc, fs), c in sorted(cells.items()):
    src = sorted((OUT / c['tag'] / 'frames').glob('*.png'))
    if len(src) < NFR:
        log(f'  {c["tag"]}: only {len(src)} frames, SKIPPED')
        continue
    wd = OUT / '_klc' / c['tag']
    subprocess.run(['rm', '-rf', str(wd)])
    wd.mkdir(parents=True)
    BAR = 28
    for i, p in enumerate(src, 1):
        rest = Image.open(p).convert('RGB')
        W, H = rest.width // 2, rest.height
        g = Image.open(GT / f'gt_{i:04d}.png').convert('RGB').resize((W, H - BAR),
                                                                    Image.LANCZOS)
        pan = Image.new('RGB', (W, H), (26, 27, 35))
        pan.paste(g, (0, BAR))
        ImageDraw.Draw(pan).text((W // 2 - 46, 8), 'GROUND TRUTH', fill=(255, 255, 255))
        comp = Image.new('RGB', (W + rest.width, H))
        comp.paste(pan, (0, 0))
        comp.paste(rest, (W, 0))
        comp.save(wd / f'{i:04d}.png')
    full = FULL / f'{c["tag"]}_720_GT_frozen_kl.mp4'
    web = WEB / f'{c["tag"]}_720.mp4'
    subprocess.run(f'ffmpeg -nostdin -v error -y -framerate 20 -i {wd}/%04d.png '
                   f'-c:v libx264 -crf 20 -preset slow -pix_fmt yuv420p '
                   f'-movflags +faststart {full}', shell=True)
    # web copy: half width and crf 32 to land near 600 KB, so 16 of them plus the
    # 3.6 MB already in the artifact stay under the 16 MB page cap.
    subprocess.run(f'ffmpeg -nostdin -v error -y -framerate 20 -i {wd}/%04d.png '
                   f'-vf scale=iw/2:ih/2 -c:v libx264 -crf 32 -preset slow '
                   f'-pix_fmt yuv420p -movflags +faststart {web}', shell=True)
    subprocess.run(['rm', '-rf', str(wd)])
    log(f'  {c["tag"]}: full {full.stat().st_size//1024} KB · '
        f'web {web.stat().st_size//1024} KB')

subprocess.run(['rm', '-rf', str(OUT / '_klc')])
res = {f'c{fc}_s{fs}': {'beta_c': beta(fc, RC), 'beta_s': beta(fs, RS),
                        'psnr': c.get('psnr'), 'ssim': c.get('ssim'),
                        'web': str(WEB / f'{c["tag"]}_720.mp4'),
                        'web_kb': (WEB / f'{c["tag"]}_720.mp4').stat().st_size // 1024
                        if (WEB / f'{c["tag"]}_720.mp4').exists() else None}
       for (fc, fs), c in sorted(cells.items())}
(OUT / 'r30kl_grid_results.json').write_text(json.dumps(res, indent=1))
log(f'total web bytes: {sum(v["web_kb"] or 0 for v in res.values())/1024:.1f} MB')
print('RESULTS_JSON_START')
print(json.dumps(res, indent=1))
print('RESULTS_JSON_END')
