"""Token-grid cells derived from the REAL training frames.

Fidelity notes:
  * the crop is the UNION silhouette box over the window, which is what
    rung27_selfattn_lora.py does before calling pipe.get_cond (fixed union crop).
  * the grid drawn is 12x12 for legibility; the real cond is 1029 DINOv3 tokens
    (32x32 patches + CLS/registers) and the figure says so in its caption.
"""
import numpy as np
from PIL import Image
from palette import mix, WHITE

FRAME = '/net/projects/ranalab/rajhansini/TRELLIS.2/data/spot_lava/frames_from_video/frame_%04d.png'


def union_bbox(idxs, thresh=245):
    U = None
    for i in idxs:
        a = np.array(Image.open(FRAME % i).convert('RGB'))
        m = a.min(axis=2) < thresh
        U = m if U is None else (U | m)
    ys, xs = np.where(U)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    sz = int(max(xs.max() - xs.min(), ys.max() - ys.min()))
    return (int(cx - sz // 2), int(cy - sz // 2), int(cx + sz // 2), int(cy + sz // 2))


def frame_cells(i, bbox, n=12):
    """-> (lum[n,n] in 0..1, obj[n,n] in 0..1)"""
    im = Image.open(FRAME % i).convert('RGB').crop(bbox)
    small = np.asarray(im.resize((n, n), Image.BOX)).astype(np.float32) / 255.0
    lum = small @ np.array([0.299, 0.587, 0.114], np.float32)
    objm = np.asarray(im.resize((n, n), Image.BOX)).min(axis=2) < 240
    # local contrast stretch inside the object so the lava structure is legible
    if objm.any():
        v = lum[objm]
        lo, hi = np.percentile(v, 4), np.percentile(v, 97)
        lum = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    return lum, objm.astype(np.float32)


def tint_cell(lum, obj, tint):
    """One cell colour: saturated tint inside the object, pale outside."""
    # inside: dark tint at low luminance -> bright tint at high luminance
    inside = mix(mix(tint, (0.06, 0.09, 0.16), 0.42), mix(tint, WHITE, 0.30), float(lum))
    outside = mix(tint, WHITE, 0.90)
    return mix(outside, inside, float(obj))


def ramp_weights(r, c, R, C):
    """Diagonal 3-colour ramp: C1 -> C2 -> C3 along the (r+c) diagonal."""
    u = (r + c) / float((R - 1) + (C - 1))
    w1 = max(0.0, 1.0 - 2.0 * u)
    w3 = max(0.0, 2.0 * u - 1.0)
    w2 = 1.0 - w1 - w3
    return w1, w2, w3
