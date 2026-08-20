"""axis_sheet.py — the meshes do not share a canonical up axis, so render each one
under all six signed axis permutations and let the eye decide which is upright.
The chosen preset is then baked into the mesh by render_hero.py --axis."""
import math, sys
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
DEVICE='cuda'; RES=340; YAW=35.0
_FX=1.0/(2.0*math.tan(math.radians(20.0)))
EXT=torch.tensor([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]],dtype=torch.float32)
INT=torch.tensor([[_FX,0,.5],[0,_FX,.5],[0,0,1.]],dtype=torch.float32)

# six ways to declare "which mesh axis points up", as 3x3 permutation/sign matrices
AXES = {
 'A_xyz' : np.eye(3),
 'B_y2z' : np.array([[1,0,0],[0,0,1],[0,-1,0]],float),
 'C_z2y' : np.array([[1,0,0],[0,0,-1],[0,1,0]],float),
 'D_x2z' : np.array([[0,0,1],[0,1,0],[-1,0,0]],float),
 'E_flipz': np.array([[1,0,0],[0,-1,0],[0,0,-1]],float),
 'F_y2z_r': np.array([[-1,0,0],[0,0,1],[0,1,0]],float),
}

def i2p(i,n,f):
    r=torch.zeros((4,4),dtype=i.dtype,device=i.device)
    r[0,0]=2*i[0,0]; r[1,1]=2*i[1,1]; r[0,2]=2*i[0,2]-1; r[1,2]=-2*i[1,2]+1
    r[2,2]=f/(f-n); r[2,3]=n*f/(n-f); r[3,2]=1.; return r

def yawR(d):
    y=math.radians(d)
    return np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])

def fit(v):
    lo,hi=v.min(0),v.max(0); return (v-(lo+hi)/2)*(0.99999/(hi-lo).max())

def pick(d):
    uv=sorted(d.glob('*_uv.obj'))
    if not uv: return sorted(d.glob('*.obj'))[:1]
    lo=[p for p in uv if any(k in p.stem for k in ('lowres','low_resolution','closed'))]
    return [lo[0] if lo else uv[0]]

import nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext()
proj=(i2p(INT.to(DEVICE),0.5,3.0)@EXT.to(DEVICE)).unsqueeze(0)
names=sys.argv[1:] or sorted(p.name for p in HERE.iterdir()
                             if p.is_dir() and p.name not in ('hero','sheet','axis'))
outd=HERE/'axis'; outd.mkdir(exist_ok=True)
for name in names:
    src=pick(HERE/name)[0]
    m=trimesh.load(src,process=False,force='mesh')
    V0=np.asarray(m.vertices,float); F=np.asarray(m.faces,np.int32)
    tiles=[]
    for tag,Ax in AXES.items():
        v=fit(V0 @ (yawR(YAW)@Ax).T)
        vt=torch.tensor(v,dtype=torch.float32,device=DEVICE)
        ft=torch.tensor(F,dtype=torch.int32,device=DEVICE).contiguous()
        vh=torch.cat([vt[None],torch.ones_like(vt[None][...,:1])],-1)
        rast,_=dr.rasterize(ctx,torch.bmm(vh,proj.transpose(-1,-2)).contiguous(),ft,(RES,RES))
        mask=(rast[0,...,3]>0)
        tri=vt[ft.long()]
        fn=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=1),dim=1)
        nrm=fn[(rast[0,...,3].long()-1).clamp(min=0)]
        def lam(dv,k):
            L=torch.tensor(dv,dtype=torch.float32,device=DEVICE); L=L/L.norm()
            return (nrm@L).clamp(min=0)*k
        sh=(0.26+lam((-0.5,-1.0,0.75),0.72)+lam((0.8,-0.6,0.15),0.20)+lam((0.0,1.0,0.35),0.16)).clamp(0,1)
        img=torch.ones(RES,RES,3,device=DEVICE)
        img=torch.where(mask[...,None], sh[...,None]*torch.tensor([.74,.745,.76],device=DEVICE), img)
        tiles.append((Image.fromarray((img.clamp(0,1).cpu().numpy()*255).astype(np.uint8)),tag))
    LAB=24; C=3; R=2
    canvas=Image.new('RGB',(RES*C,(RES+LAB)*R),(255,255,255)); d=ImageDraw.Draw(canvas)
    for i,(im,tag) in enumerate(tiles):
        r,c=divmod(i,C); x,y=c*RES,r*(RES+LAB)
        d.rectangle([x,y,x+RES-1,y+LAB-1],fill=(238,240,244)); d.text((x+8,y+6),tag,fill=(18,20,26))
        canvas.paste(im,(x,y+LAB))
    canvas.save(outd/f'AXIS_{name}.png'); print('axis sheet:',name,flush=True)
