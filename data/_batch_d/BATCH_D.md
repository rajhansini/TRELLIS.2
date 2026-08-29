# Batch D — supplementary meshes, Kling GT

11 assets over 5 meshes, all sourced from `supplementary_meshes/` (see
`experiments/dynamesh/MESH_CANDIDATES.md` for provenance). Conditioning renders came
from `supplementary_meshes/front/`, camera picked by `best_view.py` (max exposed
surface area, Kling camera: 1024 px, dist 2.6, fov 30).

| asset | mesh | mesh path | effect |
|---|---|---|---|
| `moai_silver` | moai | `data/moai_silver/mesh/moai.obj` | silver / metallic |
| `moai_animated` | moai | `data/moai_animated/mesh/moai.obj` | animated face treatment |
| `gargoyle_spiral` | gargoyle | `data/gargoyle_spiral/mesh/gargoyle.obj` | spiral / BZ target waves |
| `gargoyle_effect_one` | gargoyle | `data/gargoyle_effect_one/mesh/gargoyle.obj` | effect 1 |
| `airplane_blub` | aircraft | `data/airplane_blub/mesh/aircraft.obj` | blub |
| `airplane_red_cracks` | aircraft | `data/airplane_red_cracks/mesh/aircraft.obj` | red cracks |
| `teddy_bleach` | teddy | `data/teddy_bleach/mesh/teddy.obj` | bleach spotting |
| `teddy_fusion` | teddy | `data/teddy_fusion/mesh/teddy.obj` | fusion |
| `goat_burnt` | goat | `data/goat_burnt/mesh/goat.obj` | scorch / char |
| `goat_flower` | goat | `data/goat_flower/mesh/goat.obj` | flower |
| `goat_clay` | goat | `data/goat_clay/mesh/goat.obj` | dried clay / mud |

NAMING: the asset prefix is `airplane_*` but the mesh is `aircraft.obj` — the source
file is named aircraft. Kept as-is rather than renamed, because renaming a mesh orphans
the absolute paths every `config.json` stores.

## Upload → sort → extract

1. Upload the mp4s into `_batch_d/incoming/` (scp line in chat).
2. `bash _batch_d/sort_incoming.sh` — moves each `<asset>.mp4` to `data/<asset>/video/`.
   Filenames must match the asset names above exactly or the file is skipped, not renamed.
3. Extract frames into `data/<asset>/frames_from_video/frame_%04d.png`, the layout
   `tokens.py` and every training script read.

## Not yet verified on these meshes

Watertight / single component, and UVs — neither was checked. The texel metrics assert
identical voxel coordinates across frames, so a multi-shell mesh voids the measurement.
`aircraft` in particular has an open nose ring.
