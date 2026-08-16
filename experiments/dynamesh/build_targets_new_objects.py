"""
build_targets_new_objects.py — build gt_targets for the four solved objects.

STEP 3. solve_orientation.py found the rotation that puts each mesh where its
video is; this bakes that in and runs the existing target pipeline on top.

PER OBJECT
  1  canonical(mesh) @ R  ->  data/<obj>/mesh/<obj>_render_frame.obj
     Exactly the role spot_render_frame.obj plays: a mesh already sitting in the
     camera frame, so nothing downstream has to transform it again.
  2  make_render_mask.py --raw   ->  out/gt_targets_<obj>/render_mask.npy
     --raw is REQUIRED. That script's own preprocess() carries the wrong axis map
     ((x,-z,y) instead of the verified (-x,z,y)); --raw skips it entirely, so
     baking the pose into the .obj at step 1 sidesteps the bug rather than
     depending on it. Nothing in that file is modified.
  3  make_gt_targets.py          ->  out/gt_targets_<obj>/frames/gt_*.png
     The 2D copy: for every pixel of OUR silhouette, take the video's colour at
     that same pixel; fill the thin rim from the nearest valid neighbour.

THE BACKGROUND THRESHOLD IS COMPUTED, NOT ASSUMED
  make_gt_targets.py defaults to --bg-thresh 245, which is a property of the
  teapot and spot videos, not of video. whale_spots sits on 235 and penguin on
  249, so 245 would swallow the whole whale frame and clip the penguin. The
  threshold is instead read off each video's own border, the same rule that made
  the alignment diagnostic work on all seven objects, and is PRINTED so the
  choice is auditable rather than hidden.

  This only affects which pixels are recorded as coming straight from the video
  (`strict_*.npy`) versus filled. The target region itself is always our own
  silhouette, so a mis-set threshold cannot put background colour inside it.

121 FRAMES, NOT 150
  Every one of these videos is 121 frames. Passing --n-frames explicitly rather
  than inheriting the 150 default, which would fail on a missing frame 122.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

_HERE = Path(__file__).resolve().parent
T2 = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument('--orient', default='out/orient/orientation.json')
ap.add_argument('--suffix', default='', help='appended to the gt_targets tag, e.g. _guan')
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--objects', default='teapot_porcelain,horse_metal,'
                                     'penguin_circuits,whale_spots')
ap.add_argument('--margin', type=int, default=6,
                help='how far below the measured background level to put the '
                     'object/background cut, in 0-255 min-channel units')
A = ap.parse_args()


def canonical(v):
    """centre, uniform scale into [-0.5,0.5], canonical axis map (-x, z, y).
    Verified exact against the spot raw/canonical pair, max|d| = 5.0e-09."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = (v - (lo + hi) / 2) * (0.99999 / (hi - lo).max())
    return v[:, (0, 2, 1)] * np.array([-1.0, 1.0, 1.0])


def bg_threshold(frames_dir, n_probe=8):
    """Read the background level off the frames' own borders.

    make_gt_targets.py classifies a pixel as object when min(R,G,B) < thresh, so
    the cut has to sit BELOW the background's min-channel. Probing several frames
    rather than one, since a single frame can have the object touching an edge.
    """
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
    rep = {r['object'].split()[0]: r for r in json.load(open(_HERE / A.orient))}
    objs = [o.strip() for o in A.objects.split(',') if o.strip()]
    print('=' * 96)
    print(f'STEP 3 — build gt_targets for {len(objs)} objects at {A.res}^2')
    print('=' * 96, flush=True)

    summary = []
    for o in objs:
        r = rep[o]
        print(f'\n{"─" * 96}\n{o}   solved IoU {r["iou_after"]:.4f}\n{"─" * 96}', flush=True)

        # ---- 1. bake the pose into a render-frame mesh
        src = T2 / r['mesh']
        mesh = trimesh.load(src, process=False, force='mesh')
        V = canonical(np.asarray(mesh.vertices, dtype=np.float64).copy())
        V = V @ np.array(r['R'], dtype=np.float64).T
        F = np.asarray(mesh.faces)
        dst = src.parent / f'{o}_render_frame{A.suffix}.obj'
        trimesh.Trimesh(vertices=V, faces=F, process=False).export(dst)
        print(f'[MESH] {len(V):,}v {len(F):,}f  ->  {dst}')
        print(f'       extent {np.round(V.max(0) - V.min(0), 4)}  '
              f'centre {np.round((V.max(0) + V.min(0)) / 2, 4)}', flush=True)

        # ---- 2. our silhouette, from that exact mesh
        tag = f'gt_targets_{o}{A.suffix}'
        run([PY, _HERE / 'make_render_mask.py', '--mesh', dst,
             '--raw', '--res', A.res, '--tag', tag])

        # ---- 3. the 2D copy
        gd = T2 / r['frames']
        thr, lvl, nfr = bg_threshold(gd)
        print(f'[BG] measured background level {lvl:.1f} -> --bg-thresh {thr}   '
              f'({nfr} frames)   default 245 would have been '
              f'{"WRONG" if abs(thr - 245) > 8 else "acceptable"}', flush=True)
        run([PY, _HERE / 'make_gt_targets.py', '--gt-dir', gd,
             '--render-mask', _HERE / 'out' / tag / 'render_mask.npy',
             '--render-res', A.res, '--n-frames', nfr,
             '--bg-thresh', thr, '--tag', tag])

        m = np.load(_HERE / 'out' / tag / 'render_mask.npy')
        j = json.load(open(_HERE / 'out' / tag / 'gt_targets.json'))
        summary.append(dict(object=o, mesh=str(dst), tag=tag, n_frames=nfr,
                            bg_thresh=thr, bg_level=lvl, render_res=A.res,
                            silhouette_px=int(m.sum()), iou_align=r['iou_after'],
                            **{k: j[k] for k in ('strict_px_min', 'strict_px_max',
                                                 'iou_min', 'iou_max', 'white_px_inside')}))
        print(f'[OK] {o}: {int(m.sum()):,} px silhouette, {nfr} targets', flush=True)

    json.dump(summary, open(_HERE / 'out' / 'new_targets_summary.json', 'w'), indent=2)
    print(f'\n{"=" * 96}')
    print(f'{"object":<22}{"px":>10}{"frames":>8}{"bg":>5}  '
          f'{"from video":>22}  align')
    for s in summary:
        pc = 100 * s['strict_px_min'] / max(s['silhouette_px'], 1)
        pcx = 100 * s['strict_px_max'] / max(s['silhouette_px'], 1)
        print(f'{s["object"]:<22}{s["silhouette_px"]:>10,}{s["n_frames"]:>8}'
              f'{s["bg_thresh"]:>5}  {pc:>9.1f}% .. {pcx:>6.1f}%  {s["iou_align"]:.4f}')
    print(f'\n[SAVE] out/new_targets_summary.json\n[DONE]', flush=True)


if __name__ == '__main__':
    main()
