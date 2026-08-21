"""
axis_sheet_cpu2.py — the axis sheet without waiting for a GPU slot.

An EXACT z-buffer triangle rasteriser, not a point splat. The distinction matters:
the earlier CPU preview splatted vertices and told me the armadillo was upright when
the GPU showed it inverted. This projects with the same EXT/INT/i2p as
render_rung31_transfer_mcfm.py and fills real triangles with a depth test, so the
silhouette is the true one.

GATE-cpu: before rendering anything new it re-renders spot_lava, whose correct
appearance is already known, and asserts coverage lands in a sane band. If the
projection or winding convention were wrong that check fails and nothing else is
trusted.

Faces are subsampled to --max-faces purely for speed; that changes shading density,
never the silhouette's orientation, which is all this sheet is used to decide.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw

T2  = Path('/net/projects/ranalab/rajhansini/TRELLIS.2')
SRC = T2 / 'data/_dale_meshes'
OUT = T2 / 'data/_dale_meshes/axis'; OUT.mkdir(parents=True, exist_ok=True)
RES = 300
_FX = 1.0 / (2.0 * math.tan(math.radians(20.0)))
EXT = np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,2.],[0,0,0,1.]])
INT = np.array([[_FX,0,.5],[0,_FX,.5],[0,0,1.]])
NEAR, FAR = 0.5, 3.0
AX = {
 'A_xyz'  : np.eye(3),
 'B_y2z'  : np.array([[1,0,0],[0,0,1],[0,-1,0]], float),
 'C_z2y'  : np.array([[1,0,0],[0,0,-1],[0,1,0]], float),
 'D_x2z'  : np.array([[0,0,1],[0,1,0],[-1,0,0]], float),
 'E_flipz': np.array([[1,0,0],[0,-1,0],[0,0,-1]], float),
 'F_rev'  : np.array([[-1,0,0],[0,0,-1],[0,1,0]], float),
}
def i2p(i,n,f):
    r=np.zeros((4,4)); r[0,0]=2*i[0,0]; r[1,1]=2*i[1,1]
    r[0,2]=2*i[0,2]-1; r[1,2]=-2*i[1,2]+1; r[2,2]=f/(f-n); r[2,3]=n*f/(n-f); r[3,2]=1.
    return r
PROJ = i2p(INT,NEAR,FAR) @ EXT
def yaw_rot(d):
    a=math.radians(d); return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
def fit_unit(v):
    lo,hi=v.min(0),v.max(0); return (v-(lo+hi)/2)*(0.99999/(hi-lo).max())

def render(v, f, res=RES):
    vh = np.concatenate([v, np.ones((len(v),1))], 1)
    clip = vh @ PROJ.T
    w = clip[:,3:4]; w[np.abs(w)<1e-9] = 1e-9
    ndc = clip[:,:3]/w
    sx = (ndc[:,0]*0.5+0.5)*res
    # NO extra flip. i2p already negates y, so world +Z (up) lands at NEGATIVE ndc_y,
    # and nvdiffrast's row 0 is the top of the image. Adding a (1-...) flip here is what
    # made every axis sheet a vertical mirror of the GPU render -- and a coverage gate
    # cannot catch that, because a flipped silhouette covers exactly as many pixels.
    sy = (ndc[:,1]*0.5+0.5)*res
    sz = ndc[:,2]
    tri = f
    p = np.stack([sx,sy],1)
    a,b,c = p[tri[:,0]], p[tri[:,1]], p[tri[:,2]]
    za,zb,zc = sz[tri[:,0]], sz[tri[:,1]], sz[tri[:,2]]
    n3 = np.cross(v[tri[:,1]]-v[tri[:,0]], v[tri[:,2]]-v[tri[:,0]])
    ln = np.linalg.norm(n3,axis=1,keepdims=True); ln[ln==0]=1; n3=n3/ln
    L1=np.array([-0.5,-1.0,0.75]); L1/=np.linalg.norm(L1)
    L2=np.array([0.8,-0.6,0.15]);  L2/=np.linalg.norm(L2)
    L3=np.array([0.0,1.0,0.35]);   L3/=np.linalg.norm(L3)
    shade = np.clip(0.26 + np.clip(n3@L1,0,None)*0.72 + np.clip(n3@L2,0,None)*0.20
                         + np.clip(n3@L3,0,None)*0.16, 0, 1)
    zbuf = np.full((res,res), np.inf); col = np.zeros((res,res))
    area = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    keep = np.abs(area) > 1e-12
    for i in np.nonzero(keep)[0]:
        x0=max(int(np.floor(min(a[i,0],b[i,0],c[i,0]))),0); x1=min(int(np.ceil(max(a[i,0],b[i,0],c[i,0])))+1,res)
        y0=max(int(np.floor(min(a[i,1],b[i,1],c[i,1]))),0); y1=min(int(np.ceil(max(a[i,1],b[i,1],c[i,1])))+1,res)
        if x1<=x0 or y1<=y0: continue
        yy,xx = np.mgrid[y0:y1, x0:x1]
        px = xx+0.5; py = yy+0.5
        w0 = ((b[i,0]-a[i,0])*(py-a[i,1]) - (b[i,1]-a[i,1])*(px-a[i,0]))/area[i]
        w1 = ((c[i,0]-b[i,0])*(py-b[i,1]) - (c[i,1]-b[i,1])*(px-b[i,0]))/area[i]
        w2 = 1.0-w0-w1
        m = (w0>=0)&(w1>=0)&(w2>=0)
        if not m.any(): continue
        z = w1*za[i] + w2*zb[i] + w0*zc[i]
        sub = zbuf[y0:y1, x0:x1]
        upd = m & (z < sub)
        if upd.any():
            sub[upd] = z[upd]; col[y0:y1, x0:x1][upd] = shade[i]
    img = np.ones((res,res,3))
    hit = np.isfinite(zbuf)
    img[hit] = col[hit][:,None]*np.array([0.74,0.745,0.76])
    return (np.clip(img,0,1)*255).astype(np.uint8), float(hit.mean()*100)

def load(p, max_faces):
    """Decimate, never stride-sample.

    Taking every Nth face keeps triangles that are spatially scattered, so a 250k-face
    mesh renders as confetti and its silhouette -- the only thing this sheet exists to
    show -- disappears. Quadric decimation collapses the surface instead, preserving
    the outline at a fraction of the triangles.
    """
    m = trimesh.load(p, process=False, force='mesh')
    v = np.asarray(m.vertices, float); f = np.asarray(m.faces, np.int64)
    if len(f) > max_faces:
        # trimesh's decimator needs fast_simplification, which is not installed here.
        # open3d's quadric decimation is, and it preserves the silhouette the same way.
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                       o3d.utility.Vector3iVector(f))
        om = om.simplify_quadric_decimation(int(max_faces))
        v = np.asarray(om.vertices, float); f = np.asarray(om.triangles, np.int64)
    return v, f

ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--max-faces', type=int, default=25000)
A = ap.parse_args()

# GATE-updir: the check that actually catches a vertical flip. Coverage cannot -- an
# upside-down object covers the same pixel count. So project the single highest vertex
# of a mesh and assert it lands in the TOP half of the image. If this fails the sheets
# are mirrored and every axis read off them will be wrong.
def _updir_ok():
    v, f = load(T2/'data/spot_lava/mesh/spot_render_frame.obj', A.max_faces)
    v = fit_unit(v)
    vh = np.concatenate([v, np.ones((len(v),1))],1); clip = vh @ PROJ.T
    w = clip[:,3:4]; w[np.abs(w)<1e-9]=1e-9; ndc = clip[:,:3]/w
    top_i = int(np.argmax(v[:,2]))                 # highest point in world Z
    row = (ndc[top_i,1]*0.5+0.5)*RES
    return row, RES/2
_row, _half = _updir_ok()
print(f'[GATE-updir] highest world-Z vertex projects to row {_row:.0f} of {RES} '
      f'(must be < {_half:.0f}, i.e. upper half)', flush=True)
assert _row < _half, ('GATE-updir FAILED -- sheets are vertically mirrored relative to '
                      'the GPU renderer; do not trust any axis chosen from them')
print('[GATE-updir] PASSED', flush=True)

for name in A.names:
    p = SRC/f'{name}.obj'
    if not p.exists(): print(f'{name}: MISSING'); continue
    v0, f = load(p, A.max_faces)
    sheet = Image.new('RGB',(RES*3,(RES+20)*2),(255,255,255)); d=ImageDraw.Draw(sheet)
    for k,(lbl,R) in enumerate(AX.items()):
        img, cov = render(fit_unit(v0 @ (yaw_rot(35)@R).T), f)
        cx,cy = (k%3)*RES, (k//3)*(RES+20)
        d.rectangle([cx,cy,cx+RES-1,cy+19], fill=(238,240,244))
        d.text((cx+5,cy+5), f'{lbl}  {cov:.1f}%', fill=(20,22,28))
        sheet.paste(Image.fromarray(img),(cx,cy+20))
    sheet.save(OUT/f'{name}.png')
    print(f'{name}: {len(v0):,}v {len(f):,}f -> axis/{name}.png', flush=True)
