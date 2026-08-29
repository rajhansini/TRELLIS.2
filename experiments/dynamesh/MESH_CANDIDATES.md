# New mesh candidates — sourced from other lab repos

Found by sweeping `/net/projects/ranalab/*` (excluding our own tree) for `.obj/.glb/.ply`.
Stats are verts / triangles / normalised bbox, read off the files directly.

## Where the meshes live

| source | count | what it is |
|---|---|---|
| `share/projects/vcf/QuadWild_300_Meshes/{Organic,Mechanical}` | 133 + 168 | the richest set by far; a quad-remeshing benchmark, so shapes are clean and repaired (a `_Broken` sibling dir holds the unrepaired originals — do not use it) |
| `itailang/mesh_feature_fields/meshes` | 30 | man-made: instruments, vehicles, lamps |
| `itailang/multi_iSeg/meshes` | 29 | segmentation set; some `.off` |
| `itailang/instant-edit/data` | 26 | Text2Mesh-style: characters, animals |
| `ddecatur/latent-nerf/shapes` | 21 | the Latent-NeRF shape set |
| `ddecatur/analysis-via-synthesis/data/source_meshes` | 7 | alien, candle, lamp, person, shoe, vase, horse |
| `guanc/L4GM-official/mesh_assets` | 12 | teapot remeshing variants, mostly too coarse |

Skipped: `PointCloudWork/` is ~54k meshes but is entirely ShapeNet chairs/tables/vases
procedural dumps — no silhouette variety, not worth mining.

## Tier 1 — run these first

New object categories we do not already cover, triangle count in the range that
renders and voxelizes cleanly (5k-40k), and a silhouette that survives a figure.

| mesh | tris | category (new to us) | path |
|---|---|---|---|
| blub.obj | 14208 | fish — Keenan Crane's Blub, Spot's companion | `ddecatur/latent-nerf/shapes/blub.obj` |
| teddy.obj | 5760 | plush toy — good for fur/moss effects | `ddecatur/latent-nerf/shapes/teddy.obj` |
| robot.obj | 6132 | hard-surface character | `ddecatur/latent-nerf/shapes/robot.obj` |
| dog.obj | 6164 | quadruped | `share/projects/vcf/QuadWild_300_Meshes/Organic/dog.obj` |
| pig.obj | 6328 | quadruped | `share/projects/vcf/QuadWild_300_Meshes/Organic/pig.obj` |
| kitten.obj | 6356 | small mammal | `share/projects/vcf/QuadWild_300_Meshes/Organic/kitten.obj` |
| gargoyle.obj | 7222 | stone statue — natural fit for lava/crack | `share/projects/vcf/QuadWild_300_Meshes/Organic/gargoyle.obj` |
| beetle.obj | 7558 | insect — no insect in our set | `share/projects/vcf/QuadWild_300_Meshes/Mechanical/beetle.obj` |
| guitar.obj | 8068 | instrument | `itailang/mesh_feature_fields/meshes/guitar.obj` |
| duck.obj | 14910 | bird | `share/projects/vcf/QuadWild_300_Meshes/Organic/duck.obj` |
| dino1.obj | 18112 | dinosaur | `share/projects/vcf/QuadWild_300_Meshes/Organic/dino1.obj` |
| cello.obj | 18147 | instrument, thin neck — stress case | `itailang/mesh_feature_fields/meshes/cello.obj` |
| giraffe.obj | 18474 | tall/thin aspect (0.29,1.00,0.64) | `ddecatur/latent-nerf/shapes/giraffe.obj` |
| moai.obj | 20000 | stone head | `share/projects/vcf/QuadWild_300_Meshes/Organic/moai.obj` |
| shark.obj | 20104 | fish, long axis | `share/projects/vcf/QuadWild_300_Meshes/Organic/shark.obj` |
| goat.obj | 20868 | horned quadruped | `itailang/mesh_feature_fields/meshes/goat.obj` |
| ivysaur.obj | 22496 | creature with surface detail | `itailang/instant-edit/data/ivysaur.obj` |
| elephant.obj | 23704 | large quadruped | `share/projects/vcf/QuadWild_300_Meshes/Organic/elephant.obj` |
| aircraft.obj | 24920 | vehicle | `share/projects/vcf/QuadWild_300_Meshes/Mechanical/aircraft.obj` |
| elk.obj | 27086 | antlers — thin protrusions, stress case | `share/projects/vcf/QuadWild_300_Meshes/Organic/elk.obj` |
| pegaso.obj | 30658 | winged horse | `share/projects/vcf/QuadWild_300_Meshes/Organic/pegaso.obj` |
| car.obj | 31812 | vehicle | `itailang/mesh_feature_fields/meshes/car.obj` |

## Tier 2 — good, heavier or more niche

feline (41322) · boat (47456) · lego_minifig (53905) · mask (62656) · vulpix (74996) ·
bike (91040, very thin tubes) · napoleon (98998) · buddha (126524) · shoe (126208) ·
santa (151558) · fertility (27954, abstract) · knot (15744, abstract) · pear (21504) ·
woodenfish (34910) · homer (10202) · venus (2422) · batman_bust (5000) · potion (6654) ·
cabin (7500) · nascar (7500) · bucket (7860) · bolt (9504) · wolf (9420) · helicopter (13853) ·
truck (32725) · hat (4000) · goblet_with_top (5496) · bird (2166) · leaf (2980)

## Do not use

| mesh | why |
|---|---|
| `guanc/.../wizard_hat.obj` | 352 tris — collapses under voxelization |
| `vcf/.../Organic/cactus.obj` | 1396 tris |
| `vcf/QuadWild_300_Meshes_Broken/*` | the deliberately-broken copies |
| anything in `PointCloudWork/` | 54k procedural chairs/tables/vases, no variety |

Already in our set, skip: bunny, alien, armadillo, camel, hand, horse, spot, teapot,
vase, chair, penguin, octopus, dragon, pumpkin, skull.

## Before running any of these

Not yet checked, and each can break the pipeline:
1. **Watertight / single component** — the texel metrics assert identical voxel
   coordinates across frames; a multi-shell mesh can void that.
2. **UVs** — several of these carry no `vt`; `data/dynamesh_meshes/unwrap_uv.py` exists
   for that.
3. **Scale/orientation** — QuadWild meshes are not normalised or consistently up-axis.
