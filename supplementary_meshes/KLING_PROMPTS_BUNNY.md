# Kling prompt — bunny, silver tarnish

Input image: `MESH_CANDIDATES/bunny.png` (grey render, yaw 30, pitch 36, dist 2.83,
77% of the surface visible, object fills 28% of frame).

## Why this effect, given Itai's note

"Not a dynamic process, seemed like a texture optimization process" is the failure
mode when an effect has no causal spatial order and no irreversible direction:
patches appear everywhere at once and then sharpen, which is exactly what an
optimiser looks like converging. The effects that read as processes (mold bloom,
rust bloom, lichen) all had three things the failures lacked:

1. **Nucleation sites fixed by geometry** -- it starts where physics says it must,
   not at evenly scattered random points.
2. **An advancing front** that looks different from the settled interior.
3. **Monotonic, irreversible progress** -- area only grows, nothing un-happens.

Silver tarnish adds a fourth that no texture optimiser would ever invent: the real
**colour sequence of silver sulphide**, straw to gold to magenta-bronze to indigo to
black. The order of the colours IS the clock. A viewer can tell which frame came
first from a single still, which is the strongest possible answer to "this is not a
dynamic process."

A new lock, `THE PROCESS HAS AN ORDER`, is added for exactly this reason.

## Prompt

```
THERE IS NO LIQUID IN THIS IMAGE. Nothing is wet, nothing flows, nothing soaks, nothing pools, nothing is polished or wiped. Gravity has no effect on anything in this image.

ABSOLUTELY NOTHING DRIPS: no drips, runs, rivulets, streaks, teardrops, strands or blobs anywhere, and none hang below the chin, the paws, the tail or the belly. The lower edges of the rabbit stay razor-sharp, with pure white immediately below them.

Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BACKGROUND NEVER CHANGES: every pixel outside the rabbit's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no haze.
THE SIZE NEVER CHANGES: the rabbit is a single rigid cast-silver figurine: no breathing, no ear twitch, no nose wiggle, no head turn, no settling, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, hallmarks, inscriptions or logos anywhere on the rabbit. Only abstract pattern.
THE FORM MUST STAY READABLE: the rabbit keeps the exact sculpted lighting, shading and specular highlights of the input image. The ears, the crease between the ears, the eye, the muzzle, the haunch, the forepaws and the tail stay clearly visible in every frame, and bright un-tarnished silver still shows on the crown of the back, the outer face of the ears and the top of the haunch at all times.
THE PROCESS HAS AN ORDER: the tarnish begins ONLY in the recessed and sheltered places — the crease where the ears meet the skull, the hollow of the eye socket, under the chin, the tuck between the haunch and the body, beneath the tail and between the forepaws — because that is where the air sits still. It spreads outward from those hollows onto the open curves later. No exposed high point ever tarnishes before the hollow beside it. The tarnish only ever advances: no region brightens, cleans, lifts or reverses at any moment.
THE PATTERN GROWS, IT DOES NOT SLIDE: the tarnish never scrolls, drifts, translates, or marches across the surface in any direction. Each stain appears at a fixed point and expands outward from that point, staying anchored where it started while its edges advance.
THE EFFECT NEVER LEAKS: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, bleeds, spills or radiates into the background, and nothing is emitted into the air. The gap between the ears and the space under the chin stay flat pure white.
The only thing that changes is the colour of the surface inside the outline. Sterling silver tarnishes — a pale straw-yellow film creeps out of the ear crease, the eye hollow, under the chin and the haunch tuck, each stain widening outward from its own hollow, its centre deepening through warm gold to magenta-bronze while the advancing rim stays pale straw, the oldest cores darkening further to peacock indigo and finally to matte blue-black, the polished metal's mirror sheen dulling to a soft satin as each colour band passes over it, bright un-tarnished silver still gleaming on the open curves between the stains as new stains open in the remaining hollows and widen in turn.
```

## Two alternates in the same discipline

- **Kintsugi on a porcelain rabbit.** Two-stage, which is unmistakably temporal: hairline
  crazing propagates first and only then does gold wick along the cracks it made. The
  second stage cannot precede the first, so the ordering is self-evidencing.
- **Salt efflorescence on a cast-stone rabbit.** White crystalline bloom creeps out of the
  pores, dry, irreversible, with a fibrous advancing rim. Needs the same front-loaded
  "no liquid" block as the mud prompt, since salt implies water.

Both are unused. Tarnish is the pick because the colour sequence carries the clock.
