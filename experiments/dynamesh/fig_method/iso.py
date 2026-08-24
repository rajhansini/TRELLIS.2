import math, numpy as np

COS30, SIN30 = math.cos(math.radians(30)), 0.5

def iso_axes(s):
    ex = np.array([ COS30 * s,  SIN30 * s])
    ez = np.array([-COS30 * s,  SIN30 * s])
    ey = np.array([ 0.0,       -1.0   * s])
    return ex, ey, ez

def cube_faces(i, j, k, ex, ey, ez):
    o = i * ex + j * ey + k * ez
    top   = [o+ey, o+ex+ey, o+ex+ey+ez, o+ey+ez]
    right = [o+ex, o+ex+ey, o+ex+ey+ez, o+ex+ez]
    left  = [o+ez, o+ex+ez, o+ex+ey+ez, o+ey+ez]
    return top, right, left

def pts(p):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in p)

def render_voxels(vox, s=13.0, shade=(1.0, 0.80, 0.60), color_fn=None,
                  alpha_fn=None, stroke='#ffffff', stroke_w=0.35):
    """vox: (M,3) int ijk. Returns (svg_string, (minx,miny,maxx,maxy))."""
    ex, ey, ez = iso_axes(s)
    order = np.argsort(vox[:, 0] + vox[:, 1] + vox[:, 2], kind='stable')
    out, xs, ys = [], [], []
    for n, idx in enumerate(order):
        i, j, k = vox[idx]
        col = color_fn(i, j, k) if color_fn else (0.45, 0.55, 0.75)
        a   = alpha_fn(i, j, k) if alpha_fn else 0.75
        for face, sh in zip(cube_faces(i, j, k, ex, ey, ez), shade):
            c = tuple(min(1.0, v * sh) for v in col)
            hexc = '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in c)
            out.append(f'<polygon points="{pts(face)}" fill="{hexc}" '
                       f'fill-opacity="{a:.3f}" stroke="{stroke}" '
                       f'stroke-width="{stroke_w}" stroke-opacity="{min(1.0,a+0.15):.2f}"/>')
            for x, y in face:
                xs.append(x); ys.append(y)
    return '\n'.join(out), (min(xs), min(ys), max(xs), max(ys))
