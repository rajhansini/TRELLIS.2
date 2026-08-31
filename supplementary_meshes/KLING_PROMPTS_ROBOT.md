# Kling prompt — robot, rust

Mesh: `/net/projects/ranalab/ddecatur/3DHighlighter/data/shapes/robot.obj`
(Dale's robot, the tiger-shirt figure from 3D Highlighter / latent-paint.)
**Watertight: 3,136 verts, 6,132 tris, 0 boundary edges.**
Input image: `MESH_CANDIDATES/robot_dale.png` (yaw 330, pitch 8, dist 2.41, 70% exposed).

## Why this beats the car for this effect

The car mesh we had is Itai's `mesh_feature_fields/meshes/car.obj`, and it carries 409
open boundary edges — the white slivers on the hood and flank. Kling paints through
holes. The only watertight vehicle in the lab is `nascar.obj`, but it is a smooth
low-poly body with few panel joins, so it gives rust little to organise around.

The robot is watertight AND gives the effect a far better skeleton of nucleation sites:
**feet and ankles** (standing water, ground contact), **every joint** (shoulder, elbow,
wrist, hip, knee) where movement breaks a coating, the **torso panel seams and the
shoulder plate edges**, and the **fastener recesses**. That yields the single most
legible causal order available to us: rust creeps UP from the feet and OUT from the
joints, and the head and upper chest are the last things to go. A viewer reads the
direction without being told, which is exactly what the "looks like a texture
optimization process" note was asking for.

## Prompt

```
THERE IS NO LIQUID IN THIS IMAGE. Nothing is wet, nothing drips, nothing flows, nothing pools, nothing is washed or rinsed. No water, no rain, no droplets, no puddle. The rust is dry, powdery and already set. Gravity moves nothing in this image.

ABSOLUTELY NOTHING DRIPS: no drips, runs, rivulets, hanging streaks, teardrops or blobs anywhere, and none hang below the hands, the elbows, the hips, the knees or the feet. The lower edges of the robot stay razor-sharp, with pure white immediately below them.

Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BACKGROUND NEVER CHANGES: every pixel outside the robot's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no dust, no flakes.
THE SIZE NEVER CHANGES: the robot is a single rigid statue: it never steps, shifts weight, turns its head, bends a joint, raises an arm, flexes a finger, or settles. No deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame.
NO WRITING OF ANY KIND: there are no letters, glyphs, numerals, serial numbers, barcodes, badges, emblems or logos anywhere on the robot. Only abstract pattern.
THE FORM MUST STAY READABLE: the robot keeps the exact sculpted lighting, shading and specular highlights of the input image. The head, the eye recesses, the shoulder plates, the chest panel and its seam, the elbow and knee joints, the separated fingers and the feet stay clearly visible in every frame, and clean bare metal still shows on the head, the upper chest and the outer face of both shoulder plates at all times.
THE PROCESS HAS AN ORDER: the rust starts ONLY at the feet and ankles, where the metal meets the ground, and in the joint gaps at the knees, hips, elbows, wrists and shoulders, and along the edge of the chest panel and the shoulder plates, because those are the places that sit damp and where the coating is already broken. From there it climbs UPWARD along the shins and thighs and OUTWARD from each joint onto the smooth limb segments. No smooth mid-limb surface rusts before the joint below it, and the head and upper chest are the last surfaces touched. The rust only ever advances: no area cleans, brightens, lifts or reverses at any moment.
THE PATTERN GROWS, IT DOES NOT SLIDE: the corrosion never scrolls, drifts, translates or marches along the limbs in any direction. Each patch appears at a fixed point on the body and expands outward from that point, staying anchored where it started while its edges advance.
THE EFFECT NEVER LEAKS: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, spalls, bleeds or radiates into the background, and nothing is emitted into the air. The gaps between the arms and the torso, between the legs, and between the fingers stay flat pure white.
The only thing that changes is the colour of the surface inside the outline. The bare metal corrodes — dull orange-brown blooms opening at the ankles, the joint gaps and the panel edges, each bloom widening outward from its own point with a fine granular rim, its centre deepening to dark red-brown and then to near-black scale while a halo of paler amber powder spreads ahead of the advancing edge, the smooth grey metal around each bloom dulling and freckling into a matte pitted texture as the corrosion approaches, clean bright metal still showing on the head and upper chest as new blooms open at the remaining joints and widen in turn.
```

## Mesh provenance

Four copies exist under `ddecatur/`; three are byte-identical (md5 `ea5f96bd`):
`3DHighlighter/data/shapes/`, `latent-nerf/shapes/`,
`3d-highlighter-diffusion/metrics/control/meshes/`. The fourth,
`3d-highlighter-diffusion/data/humanoid/`, has the same counts and is also watertight.
