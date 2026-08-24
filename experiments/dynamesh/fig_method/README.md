# Method figure — MCFM + rung27

`fig_mcfm_rung27.pdf` / `.svg`. Eight schematic elements, left to right:

    token maps (f-1, f, f+1) -> temporal attention -> blended tokens
                                                        \
                              voxel tokens --------------> cross-attention
                                -> voxels after CA -> self-attention -> voxels after SA

## Build

    python3 make_figure.py
    inkscape -z -A fig_mcfm_rung27.pdf --export-text-to-path fig_mcfm_rung27.svg
    inkscape -z -e fig_mcfm_rung27.png -d 600 fig_mcfm_rung27.svg

Inkscape on this cluster is **0.92.5**, so the export flags are `-e` / `-A`,
**not** `-o`. `--export-text-to-path` is deliberate: it removes any font
dependency from the PDF, so `\includegraphics` is safe on any machine.

`_core.py` holds the shared drawing helpers and the isometric voxel renderer;
`layout.py` places the eight elements. They share one namespace, so
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

## Mesh orientation

`data/dynamesh_meshes/OBJ/spot.obj` is **rotated** — its Z-Y view is a tilted cow
and is useless here. Use `data/spot_star/mesh/spot.obj`, where Y is up and Z runs
nose to tail. The figure uses yaw 135 deg about Y, which puts the head at screen
left; `UAX = 0` is therefore the head and `UAX = 1` the rear, which is what the
cross-attention response ramp keys off.
