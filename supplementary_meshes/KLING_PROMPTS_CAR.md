# Kling prompt — car, rust

Input image: `front/car.png` (grey render, up −Z, yaw 70, pitch 30, 31.1% exposed).
Mesh: `OBJ/car.obj`. Motorcycle alternative if wanted:
`multi_iSeg/meshes/meshestotrain/bike.obj` (needs its own best-view solve).

## Why this answers the two notes at once

Itai asked for a new shape ("effect is good, but shape already used — can you do it on a
car?") and named rust. Rust also happens to be the right answer to the *other* recurring
complaint, that results "look like a texture optimization process". Rust on a vehicle has
a **causal spatial order that a viewer already knows**: it starts where water sits and
paint is broken — panel seams, the lower door edge, wheel arches, stone chips on the
sills — and bleeds outward and downward from those places. Nothing about that is
uniform refinement, which is what made the ivysaur and the Egyptian statue read as
optimisation. The `THE PROCESS HAS AN ORDER` block below is what enforces it.

Rust bloom already exists on the wingnut and the tie fighter, but the objection was to the
repeated *shape*, not the effect, and neither of those is a vehicle body with seams and
arches to organise the growth.

## Note on the mesh

`front/car.png` shows a few white slivers on the hood, roof and flank. Those are holes or
thin shells in the mesh, not shading. They are small, but Kling will paint into them, so
check the returned video for pattern leaking through the body before committing to a fit.

## Prompt

```
THERE IS NO LIQUID IN THIS IMAGE. Nothing is wet, nothing drips, nothing flows, nothing pools, nothing is washed or rinsed. No water, no rain, no droplets. The rust is dry, powdery and already set. Gravity has no effect on anything in this image.

ABSOLUTELY NOTHING DRIPS: no drips, runs, rivulets, streaks hanging in the air, teardrops or blobs anywhere, and none hang below the sills, the bumpers, the wheel arches or the mirrors. The lower edges of the car stay razor-sharp, with pure white immediately below them.

Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BACKGROUND NEVER CHANGES: every pixel outside the car's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no dust, no reflection.
THE SIZE NEVER CHANGES: the car is a single rigid parked object: the wheels never turn, the body never rocks or settles, no door, hood or panel opens, no suspension movement, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame.
NO WRITING OF ANY KIND: there are no letters, glyphs, numbers, number plates, badges, emblems, brand marks or logos anywhere on the car. Only abstract pattern.
THE FORM MUST STAY READABLE: the car keeps the exact sculpted lighting, shading and specular highlights of the input image. The wheels, the wheel arches, the side air intake, the headlight recess, the roofline and the crease running along the flank stay clearly visible in every frame, and clean unrusted paint still shows across the roof, the upper doors and the top of the hood at all times.
THE PROCESS HAS AN ORDER: the rust starts ONLY where water would sit and paint would already be broken — the seam between the panels, the lower edge of the doors and sills, the inner lip of both wheel arches, the join around the air intake, and a few stone-chip specks low on the front of the body. It creeps outward and upward from those places later. No clean panel centre rusts before the seam beside it, and the roof is the last surface to be touched. The rust only ever advances: no area cleans, brightens, lifts or reverses at any moment.
THE PATTERN GROWS, IT DOES NOT SLIDE: the corrosion never scrolls, drifts, translates or marches across the bodywork in any direction. Each patch appears at a fixed point on the body and expands outward from that point, staying anchored where it started while its edges advance.
THE EFFECT NEVER LEAKS: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, spalls, bleeds or radiates into the background, and nothing is emitted into the air. The openings of the wheel arches and the gap beneath the car stay flat pure white.
The only thing that changes is the colour of the surface inside the outline. The paint corrodes — dull orange-brown blooms opening at the panel seams, the door bottoms, the arch lips and the stone chips, each bloom widening outward from its own point with a fine granular edge, its centre deepening to dark red-brown and then to near-black scale while a halo of paler amber powder spreads ahead of the rim, the smooth paint around each bloom dulling and crazing into a matte freckled texture as the corrosion approaches, clean paint still showing across the roof and upper doors as new blooms open at the remaining seams and widen in turn.
```
