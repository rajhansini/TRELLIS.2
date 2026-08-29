# Method figure — MCFM + rung27

`fig_mcfm_rung27.pdf` / `.svg`. Two rows, serpentine, matching the system
figure's reading order:

    row 1   blended tokens  <--  TEMPORAL ATTENTION  <--  f-1  f  f+1
                   |
                   v          (blended tokens sits directly above cross-attention)
    row 2   voxel tokens --> CROSS-ATTN --> after CA --> SELF-ATTN --> after SA

Row 1 runs right-to-left and row 2 left-to-right, so the eye turns once and lands
on the block the blended tokens actually feed. The previous single-row version
needed two long curves to reach cross-attention from opposite sides.

## Build

    python3 make_figure.py
    inkscape -z -A fig_mcfm_rung27.pdf --export-text-to-path fig_mcfm_rung27.svg
    inkscape -z -e fig_mcfm_rung27.png -d 600 fig_mcfm_rung27.svg

Inkscape on this cluster is **0.92.5**, so the export flags are `-e` / `-A`,
**not** `-o`. `--export-text-to-path` is deliberate: it removes any font
dependency from the PDF, so `\includegraphics` is safe on any machine.

`_core.py` holds the shared drawing helpers and the isometric voxel renderer;
`layout.py` places the two rows. They share one namespace, so
`make_figure.py` execs them in order rather than importing.

No `trimesh` and no `cairosvg` in the default python3 here — the OBJ parser and
the surface voxeliser in `voxelize.py` are deliberately dependency-free
(numpy + PIL only).

## What is drawn from real data, and what is a device

Real:

* **Token-map structure** — luminance of frames 89 / 90 / 91 of
  `data/spot_lava/frames_from_video/`, i.e. an actual `_OFFSETS['D'] = (-1,0,+1)`
  window, cropped with the same union-silhouette box the trainer uses.
* **Voxel structure** — `data/spot_star/mesh/spot.obj` surface-voxelised at 13³
  (723 cells), isometric, painter's-algorithm depth order.

Devices, stated so nobody reads them as measurements:

* **Tint** (blue / magenta / amber) marks frame identity only. It is not in the data.
* **The diagonal ramp** in the blended map shows all three contributions at once.
  The real MCFM weights are roughly uniform over the window, not spatially varying.
* **Saturation** in the two result states encodes "did this voxel get a response",
  and is a schematic of the front-view-only supervision — not a measured field.
  **Hue** is on a separate (vertical) axis so the two cannot be confused.
* Grids are 12x12 for legibility. The real conditioning is 1029 DINOv3 tokens
  (32x32 patches + CLS / registers), D = 1024.

## Why the voxels are drawn opaque

Every panel draws the SAME cell set, opaque, with occlusion culling. Response is
carried by hue and saturation ONLY — a voxel is never encoded by fading out.

The first draft faded the cross-attention rear to `fill-opacity 0.26`. On white
paper that reads as a MISSING cube, which is the exact opposite of the figure's
claim that the voxel structure is fixed and only appearance changes. Guan circled
it. Do not reintroduce an alpha ramp on the voxels.

`voxelize_exact()` (conservative triangle-box, 13-axis SAT) replaced
`sample_surface() + voxelize()`. Random surface sampling missed 4 of 484 cells;
two of them sit at `j = 13`, the top of the head, so one horn was present and the
matching one was not. Four cells is nothing numerically and very visible in a
figure whose whole point is a complete, regular structure.

`cull_hidden()` drops the 167 voxels whose +x, +y and +z neighbours are all
occupied — in this isometric view those three cover a cube's right, top and left
faces, so nothing of it can be seen. That is what makes opaque rendering cheap.

## Mesh orientation

`data/dynamesh_meshes/OBJ/spot.obj` is **rotated** — its Z-Y view is a tilted cow
and is useless here. Use `data/spot_star/mesh/spot.obj`, where Y is up and Z runs
nose to tail. The figure uses yaw 135 deg about Y, which puts the head at screen
left; `UAX = 0` is therefore the head and `UAX = 1` the rear, which is what the
cross-attention response ramp keys off.
