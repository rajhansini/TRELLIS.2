"""hero_cpu.py — final Kling input stills, per-mesh axis baked in, CPU only.

The axis per mesh was chosen from axis/AXIS_*.png. Yaw is per mesh too: the sword
is nearly edge-on at 35 deg and covers 1.4% of frame, so it needs turning further.
No pitch term — the pipeline camera already sits above the object, and adding
pitch about X is what tipped these over in the first attempt.

Rotation is baked into <name>_hero.obj so the fixed training camera stays valid."""
import math
from pathlib import Path
import numpy as np, trimesh
from PIL import Image
from scipy.ndimage import grey_erosion, binary_closing

HERE=Path(__file__).resolve().parent
RES=960
_FX=1.0/(2.0*math.tan(math.radians(20.0)))
EXT=np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])
AX={'B_y2z':np.array([[1,0,0],[0,0,1],[0,-1,0]],float),
    'E_flipz':np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)}
CFG={'skull':('B_y2z',35),'hand':('B_y2z',35),'mushroom':('B_y2z',35),
     'wingnut':('B_y2z',35),'sword':('E_flipz',70)}

def yawR(d):
    y=math.radians(d)
    return np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
def fit(v):
    lo,hi=v.min(0),v.max(0); return (v-(lo+hi)/2)*(0.99999/(hi-lo).max())
def pick(d):
    uv=sorted(d.glob('*_uv.obj'))
    if not uv: return sorted(d.glob('*.obj'))[0]
    lo=[p for p in uv if any(k in p.stem for k in ('lowres','low_resolution','closed'))]
    return lo[0] if lo else uv[0]

out=HERE/'hero'; out.mkdir(exist_ok=True)
print(f'{"mesh":<10} {"axis":<9} {"yaw":>4} {"verts":>8} {"cover%":>7}')
for name,(axk,yaw) in CFG.items():
    src=pick(HERE/name)
    m=trimesh.load(src,process=False,force='mesh')
    V=fit(np.asarray(m.vertices,float)@(yawR(yaw)@AX[axk]).T)
    F=np.asarray(m.faces)
    trimesh.Trimesh(vertices=V,faces=F,visual=getattr(m,'visual',None),
                    process=False).export(out/f'{name}_hero.obj')
    # project + depth splat, then close gaps so the silhouette is solid
    vh=np.concatenate([V,np.ones((len(V),1))],1); cam=(EXT@vh.T).T
    z=cam[:,2].copy(); z[z<1e-6]=1e-6
    x=(_FX*cam[:,0]/z+0.5)*RES; y=(1.0-(_FX*cam[:,1]/z+0.5))*RES
    ok=(x>=0)&(x<RES)&(y>=0)&(y<RES)
    xi,yi,zi=x[ok].astype(int),y[ok].astype(int),z[ok]
    o=np.argsort(-zi); dep=np.full((RES,RES),np.nan); dep[yi[o],xi[o]]=zi[o]
    msk=binary_closing(~np.isnan(dep),structure=np.ones((5,5)),iterations=2)
    d=np.where(np.isnan(dep),np.nan,dep)
    # fill closed holes with the local depth mean so shading stays smooth
    from scipy.ndimage import generic_filter
    fillv=np.nanmean(d) if np.isfinite(np.nanmean(d)) else 2.0
    d=np.where(np.isnan(d),fillv,d)
    lo,hi=np.nanmin(d[msk]),np.nanmax(d[msk])
    sh=1.0-0.70*((d-lo)/max(hi-lo,1e-6))
    img=np.ones((RES,RES),np.float32); img[msk]=np.clip(0.24+0.66*sh[msk],0,1)
    img=grey_erosion(img,size=2)
    rgb=np.dstack([img*0.98,img*0.985,img])          # a hair cool, not pure grey
    Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).save(out/f'{name}_hero.png')
    print(f'{name:<10} {axk:<9} {yaw:>4} {len(V):>8,} {100*msk.mean():>6.1f}%')
