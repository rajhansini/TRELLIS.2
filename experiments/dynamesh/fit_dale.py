"""
fit_dale.py — put a raw mesh into the pipeline's coordinate frame.

WHY IT IS NEEDED
  data/_dale_meshes/*.obj are raw: arbitrary centre, arbitrary scale, arbitrary up-axis.
  render_rung31_transfer_mcfm.py uses ONE fixed camera and expects the object centred in
  a unit cube with world Z up. Feed it a raw mesh and it renders off-frame, on its back,
  or both -- and all four yaws are wrong together, so nothing in the figure is salvageable.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
  Rigid rotation (from the axis sheet) + centre + uniform scale. No remesh, no decimate,
  no normal recompute, no UV work. Vertex count and face list are asserted unchanged, so
  the mesh in the figure is provably the mesh that was downloaded.

  Originals are never touched: output goes to data/_dale_fitted/<name>.obj.

  A 35 deg yaw is baked in for the same reason render_hero.py bakes it -- the camera is
  fixed, so a 3/4 view has to come from the mesh. yaw000 of the transfer render then
  shows the object at 3/4, and 090/180/270 walk around from there.
"""
import json, math, sys
from pathlib import Path
import numpy as np, trimesh

T2  = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
SRC = T2 / 'data/_dale_meshes'
DST = T2 / 'data/_dale_fitted'; DST.mkdir(exist_ok=True)

AX = {
 'A_xyz'  : np.eye(3),
 'B_y2z'  : np.array([[1,0,0],[0,0,1],[0,-1,0]], float),
 'C_z2y'  : np.array([[1,0,0],[0,0,-1],[0,1,0]], float),
 'D_x2z'  : np.array([[0,0,1],[0,1,0],[-1,0,0]], float),
 'E_flipz': np.array([[1,0,0],[0,-1,0],[0,0,-1]], float),
 'F_rev'  : np.array([[-1,0,0],[0,0,-1],[0,1,0]], float),
}

def yaw_rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])

def fit_unit(v):
    lo, hi = v.min(0), v.max(0)
    return (v - (lo+hi)/2) * (0.99999 / (hi-lo).max())

def main(cfg_path):
    cfg = json.load(open(cfg_path))          # {"name": ["C_z2y", yaw_deg], ...}
    print(f'{"mesh":<26}{"axis":<9}{"yaw":>4}{"verts":>10}{"faces":>10}  extent  centre')
    for name, (axk, yaw) in cfg.items():
        p = SRC / f'{name}.obj'
        m = trimesh.load(p, process=False, force='mesh')
        v0 = np.asarray(m.vertices, float); f = np.asarray(m.faces, np.int64)
        v = fit_unit(v0 @ (yaw_rot(yaw) @ AX[axk]).T)
        assert len(v) == len(v0) and len(f) == len(m.faces), 'geometry changed -- abort'
        out = DST / f'{name}.obj'
        trimesh.Trimesh(vertices=v, faces=f, process=False).export(out)
        ext = (v.max(0) - v.min(0)).max(); ctr = np.abs((v.max(0)+v.min(0))/2).max()
        assert ext <= 1.001 and ctr < 1e-4, f'{name}: fit failed ext={ext} ctr={ctr}'
        print(f'{name:<26}{axk:<9}{yaw:>4.0f}{len(v):>10,}{len(f):>10,}  {ext:.4f}  {ctr:.1e}')

if __name__ == '__main__':
    main(sys.argv[1])
