"""
build_targets_hero.py — 2D-copy targets for objects whose mesh is already posed.

WHY THIS IS NOT build_targets_new_objects.py. That script exists to bake a solved
rotation into the mesh before building targets. These five objects need no rotation
at all: the still Kling was given is `dynamesh_meshes/PNG/<mesh>.png`, which is
byte-identical to `hero/<mesh>_hero.png`, and render_hero.py wrote the mesh it
rendered that from as `hero/<mesh>_hero.obj`. So the pose is known by construction
rather than estimated, and the orientation search — the step that misfired on four
of six objects last time — is skipped entirely.

Each object's mesh has already been copied to
  data/<obj>/mesh/<obj>_render_frame.obj
so this only does the two steps that follow.

  1  make_render_mask.py --raw   ->  out/gt_targets_<obj>/render_mask.npy
     --raw is REQUIRED: that script's own preprocess() carries the wrong axis map,
     and the pose is already baked, so it must not be transformed again.

  2  make_gt_targets.py          ->  out/gt_targets_<obj>/frames/gt_*.png
     The 2D copy: for every pixel of OUR silhouette take the video's colour at that
     same pixel, filling the thin rim from the nearest valid neighbour.

GATE-align RUNS BEFORE THE TARGETS ARE USED, NOT AFTER
  A mesh in the wrong pose still produces a full set of plausible-looking targets;
  the failure only shows up as a bad score days later. So the rasterised silhouette
  is compared against the video's own first frame and the job DIES if too much of
  our silhouette falls outside the object. Coverage, not symmetric IoU: a cast
  shadow only ever ADDS to the video's mask, so it cannot move this number, which
  is exactly why symmetric IoU wrongly failed correctly-posed objects before.

THE BACKGROUND THRESHOLD IS MEASURED, NOT ASSUMED
  make_gt_targets.py defaults to --bg-thresh 245, which is a property of the teapot
  and spot videos rather than of video in general. It is read off each clip's own
  border here, and printed, so the choice is auditable.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument('--obj', required=True)
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--margin', type=int, default=6,
                help='how far below the measured background level to cut, 0-255 min-channel')
ap.add_argument('--max-uncovered', type=float, default=10.0,
                help='%% of our silhouette allowed outside the video object before dying')
A = ap.parse_args()


def bg_threshold(frames_dir, n_probe=8):
    """Background level from the frames' own borders, over several frames — a
    single frame can have the object touching an edge."""
    fs = sorted(Path(frames_dir).glob('frame_*.png'))
    step = max(1, len(fs) // n_probe)
    levels = []
    for f in fs[::step][:n_probe]:
        a = np.asarray(Image.open(f).convert('RGB'), dtype=np.int16)
        border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
        levels.append(float(np.median(border.min(axis=1))))
    lvl = float(np.median(levels))
    return int(round(lvl)) - A.margin, lvl, len(fs)


def run(cmd):
    print('  $ ' + ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(T2))
    if r.returncode != 0:
        raise SystemExit(f'FAILED ({r.returncode}): {" ".join(str(c) for c in cmd)}')


def main():
    o = A.obj
    mesh = T2 / 'data' / o / 'mesh' / f'{o}_render_frame.obj'
    gd = T2 / 'data' / o / 'frames_from_video'
    assert mesh.exists(), f'mesh missing: {mesh}'
    assert gd.exists(), f'frames missing: {gd}'
    tag = f'gt_targets_{o}'

    print('=' * 92)
    print(f'{o} — targets at {A.res}^2 from a pre-posed hero mesh')
    print('=' * 92, flush=True)

    # ---- 1. our silhouette, from the exact mesh that produced the Kling still
    run([PY, _HERE / 'make_render_mask.py', '--mesh', mesh,
         '--raw', '--res', A.res, '--tag', tag])

    thr, lvl, nfr = bg_threshold(gd)
    print(f'[BG] measured background level {lvl:.1f} -> --bg-thresh {thr}   ({nfr} frames)   '
          f'default 245 would have been {"WRONG" if abs(thr - 245) > 8 else "acceptable"}',
          flush=True)

    # ---- GATE-align: is our silhouette actually on the object?
    m = np.load(_HERE / 'out' / tag / 'render_mask.npy').astype(bool)
    f1 = np.asarray(Image.open(sorted(gd.glob('frame_*.png'))[0])
                    .convert('RGB').resize((A.res, A.res), Image.NEAREST), np.int16)
    vid = f1.min(axis=2) < thr
    unc = 100.0 * (m & ~vid).sum() / max(m.sum(), 1)
    iou = 100.0 * (m & vid).sum() / max((m | vid).sum(), 1)
    print(f'[GATE-align] ours {m.sum():,}px  video {vid.sum():,}px  '
          f'uncovered {unc:.2f}%  IoU {iou:.1f}%', flush=True)
    if unc > A.max_uncovered:
        raise SystemExit(
            f'GATE-align FAILED: {unc:.2f}% of our silhouette is off the object '
            f'(limit {A.max_uncovered}%). The mesh is not in the video\'s pose — '
            f'building targets from it would produce a full set of wrong targets '
            f'that only fail visibly days later.')

    # ---- 2. the 2D copy
    run([PY, _HERE / 'make_gt_targets.py', '--gt-dir', gd,
         '--render-mask', _HERE / 'out' / tag / 'render_mask.npy',
         '--render-res', A.res, '--n-frames', nfr,
         '--bg-thresh', thr, '--tag', tag])

    j = json.load(open(_HERE / 'out' / tag / 'gt_targets.json'))
    rec = dict(object=o, mesh=str(mesh), tag=tag, n_frames=nfr, bg_thresh=thr,
               bg_level=lvl, render_res=A.res, silhouette_px=int(m.sum()),
               uncovered_pct=round(unc, 3), align_iou_pct=round(iou, 2),
               **{k: j[k] for k in ('strict_px_min', 'strict_px_max',
                                    'iou_min', 'iou_max', 'white_px_inside')})
    json.dump(rec, open(_HERE / 'out' / tag / 'build_summary.json', 'w'), indent=2)
    pc = 100 * rec['strict_px_min'] / max(rec['silhouette_px'], 1)
    pcx = 100 * rec['strict_px_max'] / max(rec['silhouette_px'], 1)
    print(f'\n[OK] {o}: {m.sum():,} px silhouette, {nfr} targets, '
          f'{pc:.1f}%..{pcx:.1f}% straight from video, uncovered {unc:.2f}%')
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
