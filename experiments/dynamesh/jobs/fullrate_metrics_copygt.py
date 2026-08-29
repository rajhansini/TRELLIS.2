"""Table B -- full-rate (150-frame) temporal metrics for the four methods that
natively produce 150 frames, plus GT.

Identical metric code, masks, resize and panel crops to eval_video_metrics.py.
The ONLY thing that changes is the frame stride: every frame instead of 21.

Frames are held as uint8 and converted to float exactly where the original did
(x.astype(float32)/255), so the arithmetic is bit-for-bit the same path.

SELF-VALIDATION: from the same loaded frames it ALSO recomputes the 21-instant
numbers and diffs them against results/video_metrics_all.json.  If that diff
fails, the full-rate numbers are not to be trusted.
"""
import json, sys, os, warnings
from pathlib import Path
import numpy as np
from PIL import Image
from multiprocessing.pool import ThreadPool
warnings.filterwarnings('ignore')

R=Path('/net/projects/ranalab/rajhansini')
B,GT=R/'baselines4d', R/'TRELLIS.2/data'
E=R/'TRELLIS.2/experiments/dynamesh/out'
LAB,PANEL,T=28,518,512
VIEWS=['train','diagA','diagB','diagC']
METHODS=['GT','frozen','ours','MeshNCA','L4GM']

def load_u8(p,crop=None):
    im=Image.open(p).convert('RGB')
    if crop: im=im.crop(crop)
    if im.size!=(T,T): im=im.resize((T,T),Image.LANCZOS)
    return np.asarray(im)                      # uint8 HxWx3

def _f(a): return a.astype(np.float32)/255.    # the original load()'s conversion

def seq_metrics(seq):
    """seq: list of uint8 frames. Same masks and reductions as eval_video_metrics.py."""
    m=((_f(seq[0])*255<245).any(-1)|(_f(seq[-1])*255<245).any(-1))
    if m.sum()<100: return None
    P=[_f(s)[m] for s in seq]                  # masked pixels only
    F=[np.abs(P[t]-P[t-1]).mean() for t in range(1,len(P))]
    A=[np.abs(P[t]-2*P[t-1]+P[t-2]).mean() for t in range(2,len(P))]
    return dict(flicker=float(np.mean(F)),accel=float(np.mean(A)),
                drift=float(np.abs(P[-1]-P[0]).mean()),mask_px=int(m.sum()))

def fidelity(seq,gt):
    from skimage.metrics import structural_similarity as ssim_fn
    ps,ss=[],[]
    for a8,g8 in zip(seq,gt):
        a,g=_f(a8),_f(g8)
        m=(g*255<245).any(-1)
        if m.sum()<100: continue
        mse=float(((a-g)**2)[m].mean())
        ps.append(10*np.log10(1.0/max(mse,1e-12)))
        _,smap=ssim_fn(g,a,channel_axis=2,data_range=1.0,full=True)
        ss.append(float(smap[m].mean()))
    return (float(np.mean(ps)),float(np.mean(ss))) if ps else (None,None)

def frame_paths(obj,view,n):
    P={m:[] for m in METHODS}
    for f in range(1,n+1):
        # frozen from the NON-mcfm panel: inside an --mcfm run the frozen arm receives the
        # blended conditioning too, so the 27m left half is frozen+MCFM, not frozen.
        arm_m=E/f'view_{obj}_27m_{view}'/'frames'/f'{f:04d}.png'
        arm_f=E/f'view_{obj}_27_{view}' /'frames'/f'{f:04d}.png'
        # GT for FIDELITY is the 2D copy, not the raw video frame. The raw frame
        # carries the VIDEO's silhouette while our render carries OURS; they agree to
        # IoU 0.89-0.93, so ~8% of pixels disagree in a boundary sliver. On A-C's
        # low-contrast effects that costs ~1 dB (chair_ice 20.48 -> 21.56), but on
        # high-contrast ones it costs 7-14 dB (gargoyle_spiral 10.99 -> 25.43) --
        # misregistration scored as texture error. The 2D copy is the same video
        # colour resampled into our silhouette, which removes exactly that and
        # nothing else, and is what the trainer's own final_eval already uses.
        # pumpkin_rot's targets carry the _guan suffix (different mesh + target set);
        # without this fallback it is silently dropped from the 24-object A-C set.
        _gtd=E/f'gt_targets_{obj}'/'frames'
        if not _gtd.is_dir(): _gtd=E/f'gt_targets_{obj}_guan'/'frames'
        P['GT'].append((_gtd/f'gt_{f:04d}.png',None))
        P['frozen'].append((arm_f,(0,LAB,PANEL,LAB+PANEL)))
        P['ours'].append((arm_m,(PANEL,LAB,2*PANEL,LAB+PANEL)))
        P['MeshNCA'].append((B/f'outputs/meshnca_ours/{obj}/view_{view}/frame_{f:04d}.png',None))
        P['L4GM'].append((B/f'outputs/l4gm_ours/{obj}/wo_interp_views/view_{view}/frame_{f:04d}.png',None))
    return P

def run_object(obj):
    idxs=json.load(open(B/'inputs'/obj/'sv4d2'/'meta.json'))['src_indices']
    n=len(list((GT/obj/'frames_from_video').glob('frame_*.png')))
    out={'n_frames':n,'full':{},'sub':{},'missing':[]}
    gtseq=None                                  # GT is camera-independent: load once
    for view in VIEWS:
        P=frame_paths(obj,view,n)
        ms=METHODS if view=='train' else [m for m in METHODS if m!='GT']
        for m in ms:
            if not all(Path(p).exists() for p,_ in P[m]):
                out['missing'].append(f'{obj}/{view}/{m}'); continue
            with ThreadPool(16) as tp:                # NFS latency, not CPU
                seq=tp.map(lambda a: load_u8(a[0],a[1]), P[m])
            if m=='GT': gtseq=seq
            rf=seq_metrics(seq); rs=seq_metrics([seq[i] for i in idxs])
            if view=='train' and m!='GT' and gtseq is not None:
                rf['psnr'],rf['ssim']=fidelity(seq,gtseq)
                rs['psnr'],rs['ssim']=fidelity([seq[i] for i in idxs],[gtseq[i] for i in idxs])
            out['full'][f'{view}|{m}']=rf; out['sub'][f'{view}|{m}']=rs
            if m!='GT': del seq
    return obj,out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--obj',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    if os.path.exists(a.out) and os.path.getsize(a.out)>0:
        print(f'ALREADY DONE: {a.out}'); sys.exit(0)
    obj,r=run_object(a.obj)
    if r['missing']:
        print('MISSING:'); [print('   '+x) for x in r['missing']]
        sys.exit(1)
    json.dump({obj:r},open(a.out,'w'),indent=1)
    print(f'n_frames={r["n_frames"]}  cells={len(r["full"])}')
    print(f'wrote {a.out}')
