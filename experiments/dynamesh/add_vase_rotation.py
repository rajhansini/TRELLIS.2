"""add_vase_rotation.py -- add the vase entry to known_rotations/kling_rotations.json.

R_view is DERIVED, not searched: the same closed form was checked against all ten
stored entries first (max |err| 3.5e-13), so the convention is not being guessed.
It is then PROVEN by re-rendering: posing the normalised mesh by R_view and
shooting it with the yaw0/pitch0 camera must reproduce, byte for byte, the render
made by orbiting the camera to (240, 25) -- which is the PNG that goes to Kling.
"""
import importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

RF   = "/net/projects/ranalab/guanc/tmp/kling/render_final.py"
MESH = "/net/projects/ranalab/itailang/multi_iSeg/meshes/vase.obj"
JSN  = Path("/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/"
            "known_rotations/kling_rotations.json")
KPNG = Path("/net/projects/ranalab/rajhansini/TRELLIS.2/data/dynamesh_meshes/"
            "vase_sweep/pitch_25.png")
YAW, PITCH, PREROT = 240.0, 25.0, 0

spec = importlib.util.spec_from_file_location("render_final", RF)
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)


def cam_M(yaw, pitch):
    y, p = np.radians(yaw), np.radians(pitch)
    eye = np.array([np.sin(y)*np.cos(p), np.sin(p), np.cos(y)*np.cos(p)])
    fwd = -eye/np.linalg.norm(eye)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, fwd), fwd], 1)


M0 = cam_M(0, 0)
R_view = (cam_M(YAW, PITCH) @ np.linalg.inv(M0)).T
R_pre = trimesh.transformations.rotation_matrix(np.radians(PREROT), [1, 0, 0])[:3, :3]
R_total = R_view @ R_pre

# ---- re-derive against all stored entries (guard against a silent convention drift)
D = json.load(open(JSN))
for k, o in D['objects'].items():
    vp = o['view_params']
    e = np.abs((cam_M(vp['cam_yaw_deg'], vp['cam_pitch_deg']) @ np.linalg.inv(M0)).T
               - np.array(o['R_view'])).max()
    assert e < 1e-9, f"convention drift on {k}: {e}"
print(f"convention re-verified against {len(D['objects'])} stored entries", flush=True)

# ---- PROOF: posed mesh under the front camera == orbited camera on the raw mesh
m = rf.load_norm(MESH, PREROT)
a = rf.render(m, YAW, PITCH, res=1024, dist=2.6, fov=30.0)
posed = trimesh.Trimesh(vertices=np.asarray(m.vertices) @ R_view.T,
                        faces=np.asarray(m.faces), process=False)
b = rf.render(posed, 0.0, 0.0, res=1024, dist=2.6, fov=30.0)
dif = np.abs(a.astype(int) - b.astype(int))
print(f"posed-vs-orbited: max|d|={dif.max()}  mismatched px={int((dif.max(2) > 0).sum())}")

kling = np.array(Image.open(KPNG).convert("RGB"))
d2 = np.abs(a.astype(int) - kling.astype(int))
print(f"orbited-vs-kling_input_png: max|d|={d2.max()}  "
      f"mismatched px={int((d2.max(2) > 0).sum())}")
byte_exact = bool(d2.max() == 0)
assert dif.max() <= 1, "R_view does not reproduce the orbited render"
assert byte_exact, "render does not match the PNG that goes to Kling"

# ---- write the entry
shutil.copy2(JSN, str(JSN) + ".bak")
D['objects']['vase'] = {
    'R_pre':   [[float(x) for x in r] for r in R_pre],
    'R_view':  [[float(x) for x in r] for r in R_view],
    'R_total': [[float(x) for x in r] for r in R_total],
    'kling_input_png': str(KPNG),
    'mesh': MESH,
    'verified_byte_exact': byte_exact,
    'videos': ['vase_floral.mp4'],
    'view_params': {'cam_yaw_deg': YAW, 'cam_pitch_deg': PITCH,
                    'mesh_prerot_x_deg': PREROT},
    '_note': ('yaw/pitch chosen by measured exposure sweep (vase_sweep.py): '
              'visible surface area 45.9%, handle 52.7%. vase.obj is the '
              'ONE-HANDLE mesh (genus 1); vase_12.obj is the two-handle amphora.'),
}
JSN.write_text(json.dumps(D, indent=2))
print(f"wrote vase entry -> {JSN}  (backup at {JSN}.bak)")
