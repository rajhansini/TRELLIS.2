"""render_grey.py — grey-shaded render of a mesh through the pipeline camera,
tone-matched to a reference frame so brightness is comparable by eye."""
import argparse, math
from pathlib import Path
import numpy as np, torch, trimesh
from PIL import Image
_HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--ref', help='reference frame to tone-match against')
ap.add_argument('--res', type=int, default=960)
ap.add_argument('--raw', action='store_true')
ap.add_argument('--out', required=True)
A = ap.parse_args()
DEVICE='cuda'
_FX=1.0/(2.0*math.tan(math.radians(20.0)))
EXT=torch.tensor([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]],dtype=torch.float32)
INT=torch.tensor([[_FX,0,.5],[0,_FX,.5],[0,0,1.]],dtype=torch.float32)
NEAR,FAR=0.5,3.0
def i2p(i,n,f):
    r=torch.zeros((4,4),dtype=i.dtype,device=i.device)
    r[0,0]=2*i[0,0]; r[1,1]=2*i[1,1]; r[0,2]=2*i[0,2]-1; r[1,2]=-2*i[1,2]+1
    r[2,2]=f/(f-n); r[2,3]=n*f/(n-f); r[3,2]=1.; return r
def norm(v):
    lo,hi=v.min(0),v.max(0); v=(v-(lo+hi)/2)*(0.99999/(hi-lo).max())
    t=v[:,1].copy(); v[:,1]=-v[:,2]; v[:,2]=t; return v
def main():
    import nvdiffrast.torch as dr
    m=trimesh.load(A.mesh,process=False,force='mesh')
    V=np.asarray(m.vertices,np.float64).copy()
    if not A.raw: V=norm(V)
    F=np.asarray(m.faces)
    mesh=trimesh.Trimesh(vertices=V,faces=F,process=False)
    vn=np.asarray(mesh.vertex_normals)
    ctx=dr.RasterizeCudaContext()
    v=torch.from_numpy(V).float().to(DEVICE); f=torch.from_numpy(F).int().to(DEVICE).contiguous()
    vnt=torch.from_numpy(vn).float().to(DEVICE)
    full=(i2p(INT.to(DEVICE),NEAR,FAR)@EXT.to(DEVICE)).unsqueeze(0)
    eye=-(EXT[:3,:3].T@EXT[:3,3]).to(DEVICE)
    vh=torch.cat([v,torch.ones_like(v[:,:1])],-1).unsqueeze(0)
    rast,_=dr.rasterize(ctx,torch.bmm(vh,full.transpose(-1,-2)).contiguous(),f,(A.res,A.res))
    msk=(rast[0,...,3]>0)
    n=torch.nn.functional.normalize(dr.interpolate(vnt.unsqueeze(0).contiguous(),rast,f)[0][0],dim=-1)
    p=dr.interpolate(v.unsqueeze(0).contiguous(),rast,f)[0][0]
    vd=torch.nn.functional.normalize(eye-p,dim=-1)
    lam=(n*vd).sum(-1).clamp(0,1).cpu().numpy(); msk=msk.cpu().numpy()
    g=lam.copy()
    if A.ref:
        r=np.asarray(Image.open(A.ref).convert('L'),np.float32)/255
        rm=np.asarray(Image.open(A.ref).convert('RGB'),np.float32).min(2)/255<245/255
        if r.shape[0]!=A.res:
            r=np.asarray(Image.open(A.ref).convert('L').resize((A.res,A.res),Image.LANCZOS),np.float32)/255
            rm=np.asarray(Image.open(A.ref).convert('RGB').resize((A.res,A.res),Image.LANCZOS),np.float32).min(2)/255<245/255
        src,tgt=g[msk],r[rm]
        # match median and spread, so shading reads at the same level as the video
        s=(np.percentile(tgt,95)-np.percentile(tgt,5))/max(np.percentile(src,95)-np.percentile(src,5),1e-6)
        g=(g-np.median(src))*s+np.median(tgt)
        print(f'[TONE] matched to {Path(A.ref).name}:  scale {s:.3f}')
    img=np.ones((A.res,A.res),np.float32); img[msk]=np.clip(g[msk],0,1)
    Image.fromarray((img*255).astype(np.uint8)).save(A.out)
    print(f'[SAVE] {A.out}   silhouette {int(msk.sum()):,} px')
    print(f'  mean {img[msk].mean():.3f}  p05 {np.percentile(img[msk],5):.3f}  p95 {np.percentile(img[msk],95):.3f}')
main()
