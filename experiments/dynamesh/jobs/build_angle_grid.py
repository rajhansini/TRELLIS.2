#!/usr/bin/env python3
"""Angle-picker page: elevation x yaw grid for the supplementary top view.

Every cell is labelled with the frame it actually came from. Renders in flight
show their newest frame rather than nothing, so the geometry can be judged before
the colour has finished developing; a cell at frame 150 is final.
"""
import os,glob,io,base64,json
from PIL import Image
os.chdir('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
YAWS=[0,45,90,135,180,225,270,315]; ELEVS=[30,45,60]
def cell(e,y):
    fs=sorted(glob.glob('out/view_ivysaur_petal_27g_top%dy%d/frames/*.png'%(e,y)))
    if not fs: return None,0
    p=fs[-1]; f=int(os.path.basename(p)[:-4])
    im=Image.open(p).convert('RGB')
    im=im.crop((im.width//2,int(im.height*0.09),im.width,im.height))
    g=im.convert('L'); bb=g.point(lambda v:0 if v>245 else 255).getbbox()
    if bb:
        l,t,r,b=bb; im=im.crop((max(0,l-6),max(0,t-6),min(im.width,r+6),min(im.height,b+6)))
    im.thumbnail((300,300),Image.LANCZOS)
    bg=Image.new('RGB',(300,300),'white'); bg.paste(im,((300-im.width)//2,(300-im.height)//2))
    b=io.BytesIO(); bg.save(b,'JPEG',quality=80,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode(), f
grid={}; ndone=0
for e in ELEVS:
    for y in YAWS:
        s,f=cell(e,y); grid[(e,y)]=(s,f)
        if f>=150: ndone+=1
json.dump({'%d_%d'%k:v for k,v in grid.items()}, open('/tmp/angle_grid.json','w'))
print('cells with pixels: %d/24   final (f150): %d'%(sum(1 for s,_ in grid.values() if s), ndone))
