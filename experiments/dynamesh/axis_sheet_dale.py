"""
axis_sheet_dale.py — pick the right up-axis for a raw mesh BEFORE spending GPU on it.

WHY THIS EXISTS
  data/_dale_meshes/*.obj are raw source meshes: arbitrary scale, arbitrary centre,
  arbitrary up-axis. The transfer renderer uses ONE fixed camera (world Z up,
  EXTRINSICS cam_y=(0,0,-1)), so a Y-up mesh renders on its back and all four yaws
  are wrong together. Guessing costs 4 jobs x 9 min per mesh to find out.

  So: render each candidate axis through the REAL pipeline camera and look. What the
  sheet shows is exactly what render_rung31_transfer_mcfm.py will see, because the
  extrinsics, intrinsics and fit_unit here are copied from it, not approximated.

  A CPU depth-splat preview was tried for this once and gave the wrong answer for the
  armadillo -- it looked upright and rendered inverted. Rasterise on the GPU or don't
  bother.

Usage:  python axis_sheet_dale.py mesh_a mesh_b ...      # names, no .obj
"""
import math, sys
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image, ImageDraw

T2   = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
SRC  = T2 / 'data/_dale_meshes'
OUT  = T2 / 'data/_dale_meshes/axis'; OUT.mkdir(exist_ok=True)
RES  = 300
_FX  = 1.0 / (2.0 * math.tan(math.radians(20.0)))
EXT  = torch.tensor([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])
INT  = torch.tensor([[_FX,0,.5],[0,_FX,.5],[0,0,1.]])
NEAR, FAR = 0.5, 3.0

# the six rigid frames worth testing; names match render_hero.py so notes carry over
AX = {
 'A_xyz' : np.eye(3),
 'B_y2z' : np.array([[1,0,0],[0,0,1],[0,-1,0]], float),
 'C_z2y' : np.array([[1,0,0],[0,0,-1],[0,1,0]], float),
 'D_x2z' : np.array([[0,0,1],[0,1,0],[-1,0,0]], float),
 'E_flipz': np.array([[1,0,0],[0,-1,0],[0,0,-1]], float),
 'F_rev' : np.array([[-1,0,0],[0,0,-1],[0,1,0]], float),
}

def i2p(i, n, f):
    r = torch.zeros((4,4), dtype=i.dtype, device=i.device)
    r[0,0]=2*i[0,0]; r[1,1]=2*i[1,1]; r[0,2]=2*i[0,2]-1; r[1,2]=-2*i[1,2]+1
    r[2,2]=f/(f-n); r[2,3]=n*f/(n-f); r[3,2]=1.
    return r

def fit_unit(v):
    lo, hi = v.min(0), v.max(0)
    return (v - (lo+hi)/2) * (0.99999 / (hi-lo).max())

def yaw_rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])

def main():
    import nvdiffrast.torch as dr
    dev = 'cuda'
    ctx = dr.RasterizeCudaContext()
    proj = (i2p(INT.to(dev), NEAR, FAR) @ EXT.to(dev)).unsqueeze(0)
    names = sys.argv[1:]
    for name in names:
        p = SRC / f'{name}.obj'
        if not p.exists():
            print(f'{name}: MISSING'); continue
        m = trimesh.load(p, process=False, force='mesh')
        v0 = np.asarray(m.vertices, float); f = np.asarray(m.faces, np.int32)
        cols, rows = 3, 2
        sheet = Image.new('RGB', (RES*cols, (RES+20)*rows), (255,255,255))
        d = ImageDraw.Draw(sheet)
        for k, (lbl, R) in enumerate(AX.items()):
            # 35 deg yaw so the sheet shows a 3/4 view, same as the hero stills
            v = fit_unit(v0 @ (yaw_rot(35) @ R).T)
            vt = torch.tensor(v, dtype=torch.float32, device=dev)
            ft = torch.tensor(f, dtype=torch.int32, device=dev).contiguous()
            vh = torch.cat([vt[None], torch.ones_like(vt[None][...,:1])], -1)
            clip = torch.bmm(vh, proj.transpose(-1,-2)).contiguous()
            rast,_ = dr.rasterize(ctx, clip, ft, (RES,RES))
            mask = rast[0,...,3] > 0
            tri = vt[ft.long()]
            fn = torch.nn.functional.normalize(
                torch.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0], dim=1), dim=1)
            nrm = fn[(rast[0,...,3].long()-1).clamp(min=0)]
            def lam(dv,s):
                L = torch.tensor(dv, dtype=torch.float32, device=dev); L = L/L.norm()
                return (nrm @ L).clamp(min=0)*s
            sh = (0.26 + lam((-0.5,-1.0,0.75),0.72) + lam((0.8,-0.6,0.15),0.20)
                       + lam((0.0,1.0,0.35),0.16)).clamp(0,1)
            img = torch.ones(RES,RES,3, device=dev)
            img = torch.where(mask[...,None],
                              sh[...,None]*torch.tensor([.74,.745,.76], device=dev), img)
            a = (img.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            cx, cy = (k % cols)*RES, (k//cols)*(RES+20)
            d.rectangle([cx,cy,cx+RES-1,cy+19], fill=(238,240,244))
            cov = 100.0*float(mask.float().mean())
            d.text((cx+5,cy+5), f'{lbl}  {cov:.1f}%', fill=(20,22,28))
            sheet.paste(Image.fromarray(a), (cx, cy+20))
        sheet.save(OUT / f'{name}.png')
        print(f'{name}: {len(v0):,}v {len(f):,}f -> axis/{name}.png', flush=True)

if __name__ == '__main__':
    main()
