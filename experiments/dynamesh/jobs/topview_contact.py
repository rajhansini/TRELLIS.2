#!/usr/bin/env python3
"""Contact sheet of the top-view yaw sweep at frame 150, for picking the camera.

Each render is the renderer's own frozen|adapted panel, so the right half is our
result. We crop to that half: the figure only ever shows the adapted side.
"""
import os,glob
from PIL import Image
E='/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh'
os.chdir(E)
YAWS=[0,45,90,135,180,225,270,315]; F=150
tiles=[]
for y in YAWS:
    p='out/view_ivysaur_petal_27g_top60y%d/frames/%04d.png'%(y,F)
    if not os.path.exists(p): tiles.append((y,None)); continue
    im=Image.open(p).convert('RGB')
    im=im.crop((im.width//2,0,im.width,im.height))   # right half = adapted
    tiles.append((y,im))
have=[t for t in tiles if t[1]]
if not have: raise SystemExit('no frames yet')
w,h=have[0][1].size; sc=320/w
cols=4; rows=(len(tiles)+cols-1)//cols
W,H=int(w*sc),int(h*sc)
sheet=Image.new('RGB',(cols*W,rows*(H+22)),'white')
from PIL import ImageDraw
d=ImageDraw.Draw(sheet)
for i,(y,im) in enumerate(tiles):
    cx,cy=(i%cols)*W,(i//cols)*(H+22)
    if im: sheet.paste(im.resize((W,H)),(cx,cy+22))
    d.text((cx+6,cy+6),'yaw %d'%y,fill='black')
sheet.save('out/topview_contact.png')
print('wrote out/topview_contact.png  (%d/%d views)'%(len(have),len(tiles)))
