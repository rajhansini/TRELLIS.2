# supplementary_meshes

22 meshes pulled from other lab repos (see `experiments/dynamesh/MESH_CANDIDATES.md`
for where each came from), plus one front-view conditioning render per mesh.

    OBJ/            the meshes (21 -- see "rejected" below)
    front/          1024x1024 front-view render per mesh  <- feed these to the video model
    front_views.tsv chosen up-axis / yaw / pitch / exposed-area fraction per mesh
    front_sheet.png all renders on one sheet, for review
    orient/         per-mesh sheet of all six up-axis assignments
    best_view.py    the view search
    orient_sheet.py the orientation sheets

## How the view was chosen

The camera is copied from `guanc/tmp/kling/render_final.py` -- 1024 px, dist 2.6,
fov 30, same two-light rig -- so `front/*.png` are drop-in for the Kling pipeline.

The view maximises **exposed surface area**: the fraction of the mesh's own area that
is unoccluded AND front-facing. Points are sampled on the faces area-weighted,
projected, and z-buffered; survivors/total is then the visible area fraction directly,
and it accounts for self-occlusion that a normals-only estimate would miss.

Not silhouette area: silhouette rewards a broadside view of a flat object, whereas the
video model can only invent texture for surface it can actually see.

**Up-axis is fixed by hand, not searched.** Searching over up-axis as well maximised
exposure by laying animals on their backs -- a belly-up dog exposes more surface than a
standing one, and that is not a front view. The 21 up-axes in `UP_FIX` (best_view.py)
were read off `orient/*.png` by eye. Yaw is searched over the full 360 deg at 10 deg
steps; pitch over 5/10/20/30 deg, which keeps the camera near-horizontal while giving
flat subjects (aircraft, car, shark) enough headroom.

## Rejected

`rejected/elk.obj` -- the file is named elk but the geometry is an abstract sculpture
(a torus with spheres), not an elk. Dropped from the set.

## Meshes whose FRAMING is fine but whose POSE is awkward

The view search cannot fix the mesh's own pose. These four are usable but will not
give a clean side profile:

  elephant  the source mesh is rearing / reclining, not standing
  dino1     balanced on one leg
  dog       a blobby stylised dog; reads as a crouching lump at any yaw
  pig       lies horizontal; no standing pose exists in the mesh

## Still unverified before any training run

1. Watertight / single component -- the texel metrics assert identical voxel
   coordinates across frames, and a multi-shell mesh voids that. `beetle` and `car`
   visibly have open windows.
2. UVs -- several carry no `vt`. See `data/dynamesh_meshes/unwrap_uv.py`.
3. Scale/orientation are normalised only inside these scripts, not in the OBJ files.
