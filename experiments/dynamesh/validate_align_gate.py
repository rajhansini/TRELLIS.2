"""validate_align_gate.py -- is the CPU gate itself correct?

Runs the same silhouette/IoU code on objects whose rotation was already SOLVED
against their video by solve_orientation.py, with the IoU it achieved recorded in
orientation.json. If the gate reproduces those numbers it is trustworthy and a bad
score on the vase means a bad pose. If it does not, the gate is the bug and the
vase verdict is void.
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_vase_align_cpu import canonical, silhouette, video_mask, T2, E

REF = json.load(open(E/'out/orient/orientation.json'))
print(f"{'object':24s} {'recorded':>9s} {'gate IoU':>9s} {'outside%':>9s}  verdict")
for r in REF:
    name = r['object'].split()[0]
    fr = T2/r['frames']/'frame_0001.png'
    if not fr.exists():
        print(f"{name:24s} {'-':>9s}  no frames"); continue
    m = trimesh.load(T2/r['mesh'], process=False, force='mesh')
    V = canonical(np.asarray(m.vertices, float).copy()) @ np.array(r['R']).T
    ours = silhouette(V, np.asarray(m.faces))
    vid, _ = video_mask(fr)
    iou = (ours & vid).sum()/(ours | vid).sum()
    out = (ours & ~vid).sum()/max(ours.sum(), 1)
    d = iou - r['iou_after']
    print(f"{name:24s} {r['iou_after']:9.4f} {iou:9.4f} {100*out:8.2f}%  "
          f"{'MATCH' if abs(d) < 0.06 else f'OFF by {d:+.3f}'}")
