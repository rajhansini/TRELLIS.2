"""render_all_cpu.py — one solid render per mesh, CPU only, no slurm queue.

Vertex splatting left holes because vertices are sparse relative to 960^2. Sampling
POINTS ON THE FACES instead (trimesh.sample_surface) makes coverage a function of
sample count, not tessellation, so the silhouette fills in solid. Shading uses the
sampled face's normal, so curvature reads the same as a rasterised render.

Frames:
  odedstein meshes are in their own object frame -> axis + yaw baked here.
  existing assets are ALREADY fitted to the pipeline camera -> identity, front view.
"""
import math, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image
from scipy.ndimage import binary_closing, grey_erosion

HERE=Path(__file__).resolve().parent
RES=960; NPTS=1_200_000
_FX=1.0/(2.0*math.tan(math.radians(20.0)))
EXT=np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])
B=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)          # Y-up -> Z-up
E=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def yawR(d):
    y=math.radians(d)
    return np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
# odedstein: (axis, yaw).  everything else: already camera-fitted -> identity
ODED={'skull':(B,35),'hand':(B,35),'mushroom':(B,35),'wingnut':(B,35),'sword':(E,70),
      'nefertiti':(B,35),'armadillo':(B,35),'lionstatue':(B,35),'falconstatue':(B,35)}

def fit(v):
    lo,hi=v.min(0),v.max(0); return (v-(lo+hi)/2)*(0.99999/(hi-lo).max())

def pickmesh(d):
    uv=sorted(d.glob('*_uv.obj'))
    if uv:
        lo=[p for p in uv if any(k in p.stem for k in ('lowres','low_resolution','closed'))]
        return lo[0] if lo else uv[0]
    c=sorted(list(d.glob('*.obj'))+list(d.glob('*.ply')))
    c=[p for p in c if '_uv' not in p.stem]
    return c[0] if c else None

names=sys.argv[1:] or sorted(p.name for p in HERE.iterdir()
                             if p.is_dir() and p.name not in ('hero','axis','render'))
out=HERE/'render'; out.mkdir(exist_ok=True)
print(f'{"asset":<22} {"mesh":<34} {"cover%":>7}')
for name in names:
    src=pickmesh(HERE/name)
    if src is None: print(f'{name:<22} -- no mesh --'); continue
    m=trimesh.load(src,process=False,force='mesh')
    V=np.asarray(m.vertices,float); F=np.asarray(m.faces)
    if name in ODED:
        Ax,yw=ODED[name]; V=fit(V@(yawR(yw)@Ax).T)
    else:
        V=fit(V)                                   # already in camera frame
    mm=trimesh.Trimesh(vertices=V,faces=F,process=False)
    pts,fidx=trimesh.sample.sample_surface(mm,NPTS)
    fn=mm.face_normals[fidx]

    vh=np.concatenate([pts,np.ones((len(pts),1))],1); cam=(EXT@vh.T).T
    z=cam[:,2].copy(); z[z<1e-6]=1e-6
    x=(_FX*cam[:,0]/z+0.5)*RES; y=(1.0-(_FX*cam[:,1]/z+0.5))*RES
    ok=(x>=0)&(x<RES)&(y>=0)&(y<RES)
    xi,yi,zi,nn=x[ok].astype(int),y[ok].astype(int),z[ok],fn[ok]
    o=np.argsort(-zi)                              # far first; nearer overwrites
    dep=np.full((RES,RES),np.nan); nrm=np.zeros((RES,RES,3))
    dep[yi[o],xi[o]]=zi[o]; nrm[yi[o],xi[o]]=nn[o]
    msk=~np.isnan(dep)
    msk=binary_closing(msk,structure=np.ones((3,3)),iterations=1)

    def lam(d,k):
        L=np.array(d,float); L/=np.linalg.norm(L)
        return np.clip(nrm@L,0,None)*k
    sh=np.clip(0.26+lam((-0.5,-1.0,0.75),0.72)+lam((0.8,-0.6,0.15),0.20)
                    +lam((0.0,1.0,0.35),0.16),0,1)
    img=np.ones((RES,RES),np.float32); img[msk]=sh[msk]
    img=grey_erosion(img,size=2)
    rgb=np.dstack([img*0.975,img*0.982,img])
    Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).save(out/f'{name}.png')
    print(f'{name:<22} {src.name:<34} {100*msk.mean():>6.1f}%')
