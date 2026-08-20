"""axis_sheet_cpu.py — same question as axis_sheet.py, no GPU.
Which signed axis permutation makes each mesh stand upright? A depth-shaded
vertex splat through the pipeline camera is enough to judge that by eye, and it
runs on CPU in seconds instead of waiting on the slurm queue."""
import math, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
RES=300; YAW=35.0
_FX=1.0/(2.0*math.tan(math.radians(20.0)))
EXT=np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])

AXES={
 'A_xyz'  : np.eye(3),
 'B_y2z'  : np.array([[1,0,0],[0,0,1],[0,-1,0]],float),
 'C_z2y'  : np.array([[1,0,0],[0,0,-1],[0,1,0]],float),
 'D_x2z'  : np.array([[0,0,1],[0,1,0],[-1,0,0]],float),
 'E_flipz': np.array([[1,0,0],[0,-1,0],[0,0,-1]],float),
 'F_y2z_r': np.array([[-1,0,0],[0,0,1],[0,1,0]],float),
}
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

def splat(v):
    """camera-space project + depth-shaded splat -> RES x RES uint8"""
    vh=np.concatenate([v,np.ones((len(v),1))],1)
    cam=(EXT@vh.T).T                      # camera space
    z=cam[:,2].copy(); z[z<1e-6]=1e-6
    x=(_FX*cam[:,0]/z+0.5)*RES
    y=(1.0-(_FX*cam[:,1]/z+0.5))*RES
    ok=(x>=0)&(x<RES)&(y>=0)&(y<RES)
    xi,yi,zi=x[ok].astype(int),y[ok].astype(int),z[ok]
    buf=np.full((RES,RES),np.inf)
    order=np.argsort(-zi)                 # far first, near overwrites
    np.maximum.at(buf,(yi[order],xi[order]),0)   # mark coverage
    depth=np.full((RES,RES),np.nan)
    depth[yi[order],xi[order]]=zi[order]
    m=~np.isnan(depth)
    img=np.ones((RES,RES),np.float32)
    if m.any():
        d=depth[m]; lo,hi=d.min(),d.max()
        shade=1.0-0.72*((d-lo)/max(hi-lo,1e-6))   # near = bright
        img[m]=0.20+0.68*shade
    # thicken: any pixel with a neighbour covered gets filled, closes splat gaps
    from scipy.ndimage import grey_erosion
    img=grey_erosion(img,size=3)
    return (np.clip(img,0,1)*255).astype(np.uint8)

names=sys.argv[1:] or sorted(p.name for p in HERE.iterdir()
                             if p.is_dir() and p.name not in ('hero','sheet','axis'))
out=HERE/'axis'; out.mkdir(exist_ok=True)
for name in names:
    src=pick(HERE/name)[0]
    V0=np.asarray(trimesh.load(src,process=False,force='mesh').vertices,float)
    tiles=[(Image.fromarray(splat(fit(V0@(yawR(YAW)@Ax).T))).convert('RGB'),tag)
           for tag,Ax in AXES.items()]
    LAB=22; C,R=3,2
    cv=Image.new('RGB',(RES*C,(RES+LAB)*R),(255,255,255)); d=ImageDraw.Draw(cv)
    for i,(im,tag) in enumerate(tiles):
        r,c=divmod(i,C); x,y=c*RES,r*(RES+LAB)
        d.rectangle([x,y,x+RES-1,y+LAB-1],fill=(236,239,243)); d.text((x+7,y+5),tag,fill=(18,20,26))
        cv.paste(im,(x,y+LAB))
    cv.save(out/f'AXIS_{name}.png'); print('axis sheet:',name,flush=True)
