"""
yaw_sweep.py — pick the azimuth that shows a mesh best, before committing GPU to it.

The axis sheet fixes which way is UP; this fixes which way the object FACES. Both matter:
furnature_chair at the default 35 deg shows mostly the back panel, and animals_bob shows a
flat profile. Frame coverage alone is a poor guide -- a chair seen dead-on covers a lot and
reads as a rectangle -- so the sheet prints coverage but the choice is made by eye.

Same projection and shading as render_rung31_transfer_mcfm.py, and the same GATE-updir the
axis sheets use, so what appears here is what the transfer render will produce.
"""
import argparse, json, math
import numpy as np, trimesh
from PIL import Image, ImageDraw
from pathlib import Path

T2  = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
SRC = T2/'data/_dale_meshes'
OUT = T2/'data/_dale_meshes/axis'; OUT.mkdir(exist_ok=True)
RES = 300
_FX = 1.0/(2.0*math.tan(math.radians(20.0)))
EXT = np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])
INT = np.array([[_FX,0,.5],[0,_FX,.5],[0,0,1.]])
AX = {'A_xyz':np.eye(3),
      'B_y2z':np.array([[1,0,0],[0,0,1],[0,-1,0]],float),
      'C_z2y':np.array([[1,0,0],[0,0,-1],[0,1,0]],float),
      'D_x2z':np.array([[0,0,1],[0,1,0],[-1,0,0]],float),
      'E_flipz':np.array([[1,0,0],[0,-1,0],[0,0,-1]],float),
      'F_rev':np.array([[-1,0,0],[0,0,-1],[0,1,0]],float)}
def i2p(i,n,f):
    r=np.zeros((4,4)); r[0,0]=2*i[0,0]; r[1,1]=2*i[1,1]
    r[0,2]=2*i[0,2]-1; r[1,2]=-2*i[1,2]+1; r[2,2]=f/(f-n); r[2,3]=n*f/(n-f); r[3,2]=1.
    return r
PROJ=i2p(INT,0.5,3.0)@EXT
def yaw_rot(d):
    a=math.radians(d); return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
def fit_unit(v):
    lo,hi=v.min(0),v.max(0); return (v-(lo+hi)/2)*(0.99999/(hi-lo).max())
def render(v,f,res=RES):
    vh=np.concatenate([v,np.ones((len(v),1))],1); clip=vh@PROJ.T
    w=clip[:,3:4]; w[np.abs(w)<1e-9]=1e-9; ndc=clip[:,:3]/w
    sx=(ndc[:,0]*0.5+0.5)*res; sy=(ndc[:,1]*0.5+0.5)*res; sz=ndc[:,2]
    p=np.stack([sx,sy],1); a,b,c=p[f[:,0]],p[f[:,1]],p[f[:,2]]
    za,zb,zc=sz[f[:,0]],sz[f[:,1]],sz[f[:,2]]
    n3=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    ln=np.linalg.norm(n3,axis=1,keepdims=True); ln[ln==0]=1; n3=n3/ln
    def L(d,s):
        d=np.array(d,float); d/=np.linalg.norm(d); return np.clip(n3@d,0,None)*s
    sh=np.clip(0.26+L((-0.5,-1.0,0.75),.72)+L((0.8,-0.6,0.15),.20)+L((0.0,1.0,0.35),.16),0,1)
    zbuf=np.full((res,res),np.inf); col=np.zeros((res,res))
    ar=(b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    for i in np.nonzero(np.abs(ar)>1e-12)[0]:
        x0=max(int(min(a[i,0],b[i,0],c[i,0])),0); x1=min(int(max(a[i,0],b[i,0],c[i,0]))+2,res)
        y0=max(int(min(a[i,1],b[i,1],c[i,1])),0); y1=min(int(max(a[i,1],b[i,1],c[i,1]))+2,res)
        if x1<=x0 or y1<=y0: continue
        yy,xx=np.mgrid[y0:y1,x0:x1]; px=xx+0.5; py=yy+0.5
        w0=((b[i,0]-a[i,0])*(py-a[i,1])-(b[i,1]-a[i,1])*(px-a[i,0]))/ar[i]
        w1=((c[i,0]-b[i,0])*(py-b[i,1])-(c[i,1]-b[i,1])*(px-b[i,0]))/ar[i]
        w2=1-w0-w1; m=(w0>=0)&(w1>=0)&(w2>=0)
        if not m.any(): continue
        z=w1*za[i]+w2*zb[i]+w0*zc[i]; sub=zbuf[y0:y1,x0:x1]; u=m&(z<sub)
        if u.any(): sub[u]=z[u]; col[y0:y1,x0:x1][u]=sh[i]
    img=np.ones((res,res,3)); hit=np.isfinite(zbuf)
    img[hit]=col[hit][:,None]*np.array([0.74,0.745,0.76])
    return (np.clip(img,0,1)*255).astype(np.uint8), float(hit.mean()*100)

ap=argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('axis')
ap.add_argument('--max-faces',type=int,default=25000)
A=ap.parse_args()
m=trimesh.load(SRC/f'{A.name}.obj',process=False,force='mesh')
v0=np.asarray(m.vertices,float); f=np.asarray(m.faces,np.int64)
if len(f)>A.max_faces:
    import open3d as o3d
    om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v0),o3d.utility.Vector3iVector(f))
    om=om.simplify_quadric_decimation(int(A.max_faces))
    v0=np.asarray(om.vertices,float); f=np.asarray(om.triangles,np.int64)
YAWS=[0,25,45,70,90,115,135,160,180,205,225,250,270,295,315,340]
C=4; R=(len(YAWS)+C-1)//C
sheet=Image.new('RGB',(RES*C,(RES+20)*R),(255,255,255)); d=ImageDraw.Draw(sheet)
best=(0,None)
for k,y in enumerate(YAWS):
    img,cov=render(fit_unit(v0@(yaw_rot(y)@AX[A.axis]).T),f)
    if cov>best[0]: best=(cov,y)
    x,yy=(k%C)*RES,(k//C)*(RES+20)
    d.rectangle([x,yy,x+RES-1,yy+19],fill=(238,240,244))
    d.text((x+5,yy+5),f'yaw {y}   {cov:.1f}%',fill=(20,22,28))
    sheet.paste(Image.fromarray(img),(x,yy+20))
sheet.save(OUT/f'YAW_{A.name}.png')
print(f'{A.name} [{A.axis}] -> axis/YAW_{A.name}.png   max coverage {best[0]:.1f}% at yaw {best[1]}')
