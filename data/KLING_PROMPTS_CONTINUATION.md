# Kling prompts — CONTINUATION (textured start, travelling texture)

Round 1 fed Kling a **grey render** and asked it to paint. These prompts feed it the
**already-textured render** and ask the texture to *travel*. The mesh is still frozen
and the material is still the same material, but the pattern now moves a real distance
across a fixed silhouette instead of arriving onto an empty one.

Input image for each: `experiments/dynamesh/out/KLING_START/<obj>_r27_mcfmv2_D_last.png`
(rung27 + MCFM v2_D at the training view, "ours" panel, label bar cropped).

## v2 — why the first draft was wrong

Draft 1 kept coverage constant by making patches **trade in place**: one spreads, another
pulls back. That reads as pulsation, not motion, and it is the degenerate answer — near-zero
displacement is precisely what the drift guard in the metrics section is built to detect. A
texture that only breathes scores well on flicker and acceleration while going nowhere.

v2 keeps coverage constant by **inflow and outflow** instead. Material enters at one end of a
named path, travels along it, and leaves at the other. Total coverage holds *because* the
conveyor is balanced, not because nothing moves.

## Locks

Four DALE locks carry over verbatim: camera, background, size, no-writing, no-leak.
Two were written for a grey start and are replaced; four are new.

| lock | job |
|---|---|
| THE SURFACE IS ALREADY FINISHED | stops Kling rewinding to a clean object and replaying the birth sequence |
| THE PATTERN TRAVELS | names a path, a direction and a distance, so motion is directed rather than jittery |
| MATERIAL ENTERS AND LEAVES | holds coverage constant *while* everything moves |
| THE SHAPES STAY TRACKABLE | keeps blob identity so the eye can follow one across the clip |
| THE LIGHTING DOES NOT TRAVEL | shading and highlights stay pinned to the mesh; only the pattern slides over them. This is what separates texture-on-a-fixed-object from a scrolling image |

---

## 1. pumpkin_rot — mold marching around the ribs

Start frame is fully rotted: blackened lobes, pale grey-green mold patches, dark brown ribs,
no fresh orange. Path: the pale bands march **around** the pumpkin, lobe to lobe, so each one
crosses several ribs over the clip. Circumferential travel is the most legible motion this
shape offers from the training view.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BACKGROUND NEVER CHANGES: every pixel outside the pumpkin's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE SIZE NEVER CHANGES: the pumpkin is a single rigid object: no sagging, collapsing, shrinking, slumping, no rotation, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. The pumpkin itself never spins — only the pattern on it moves.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the pumpkin. Only abstract pattern.
THE SURFACE IS ALREADY FINISHED: the pumpkin in the input image is already fully rotted and that is the starting state. It never rewinds, never heals, never returns to fresh orange skin, and the rot never begins again from a clean surface. There is no before; the decay is already complete and now it moves.
THE PATTERN TRAVELS: the pale mold does not sit still and does not merely pulse. Every mold patch migrates steadily sideways around the pumpkin, lobe to lobe, crossing the raised ribs and the deep grooves between them, all of them drifting the same way around the ring. Over the clip a patch that starts on one lobe ends several lobes away. This is a slow continuous march, not a flicker, not a throb, not a swap between two states.
MATERIAL ENTERS AND LEAVES: fresh pale mold seeps into view at the trailing edge of the surface and old mold is swallowed back into the black rot at the leading edge, so exactly as much of the pumpkin is pale at the end as at the beginning. Coverage stays constant because the flow is balanced, not because anything stops moving. The pumpkin never turns uniformly black and never turns uniformly pale.
THE SHAPES STAY TRACKABLE: each mold patch keeps its own distinct outline as it travels — a shape you could point at in the first frame and still recognise many frames later, deforming slowly as it rides over the ribs but never dissolving into noise, never teleporting, never blinking out and reappearing somewhere else.
THE LIGHTING DOES NOT TRAVEL: the pumpkin's sculpted shading, its cast shadows in the grooves and its specular highlights are painted onto the mesh and stay exactly where they are in the input image. Only the mold pattern slides over them. The bright side stays the bright side and the dark grooves stay dark, even as pale patches pass through them.
THE FORM MUST STAY READABLE: the ribbed lobes, the deep grooves between them and the pale dry stem stay clearly visible in every frame, and the pumpkin is never flattened into a solid dark silhouette.
THE EFFECT NEVER LEAKS: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, bleeds, spills or radiates into the background, and nothing is emitted into the air.
The only thing that changes is the colour of the surface inside the outline. Grey-green mold migrates around the blackened rotted pumpkin — fuzzy pale colonies crawling sideways from lobe to lobe, riding up over each raised rib and down into the groove beyond it, stretching as they cross a ridge and compacting as they settle into a hollow, black rot closing in behind each one as fresh pale growth creeps into view ahead of it.
```

## 2. hand_rorschach — ink running out along the fingers

Start frame is a grey hand carrying large black blots over roughly half the surface. Path:
the blots run **wrist → knuckles → out along each finger → off the fingertips**, with new ink
entering at the cut wrist. Anatomical channelling makes the travel unmistakable.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BACKGROUND NEVER CHANGES: every pixel outside the hand's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE SIZE NEVER CHANGES: the hand is a single rigid sculpture: no finger movement, no flexing, twitching, curling, spreading, breathing, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. The hand itself never moves — only the pattern on it moves.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the hand. Only abstract pattern.
THE SURFACE IS ALREADY FINISHED: the hand in the input image is already inked and that is the starting state. It never rewinds, never wipes clean, never returns to a bare grey hand, and the ink never begins again from an unmarked surface. There is no before; the blots are already there and now they travel.
THE PATTERN TRAVELS: the ink does not sit still and does not merely pulse. Every blot runs steadily up the hand — off the wrist, across the back of the hand, over the knuckles, and out along the fingers toward the tips — all of them moving the same way. Over the clip a blot that starts near the wrist reaches the fingertips. This is a slow continuous run, not a flicker, not a throb, not a swap between two states.
MATERIAL ENTERS AND LEAVES: new black ink wells up at the cut wrist and feeds onto the hand while the ink that reaches the fingertips thins away to bare grey and is gone, so exactly as much of the hand is black at the end as at the beginning. Coverage stays constant because the flow is balanced, not because anything stops moving. The hand never fills in to solid black and never clears to plain grey.
THE SHAPES STAY TRACKABLE: each blot keeps its own distinct ragged outline as it travels — a shape you could point at in the first frame and still recognise many frames later, stretching as it is drawn out along a finger and narrowing between the knuckles, but never dissolving into noise, never teleporting, never blinking out and reappearing somewhere else.
THE LIGHTING DOES NOT TRAVEL: the hand's sculpted shading, its soft shadows between the fingers and its specular highlights are painted onto the mesh and stay exactly where they are in the input image. Only the ink slides over them. The lit side of each finger stays lit and the shadowed gaps stay shadowed, even as black blots pass through them.
THE FORM MUST STAY READABLE: the four fingers, the thumb, the knuckles and the cut wrist stay clearly visible and separable in every frame, and the hand is never flattened into a solid black silhouette.
THE EFFECT NEVER LEAKS: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, bleeds, spills or radiates into the background, and nothing runs off the fingertips into the air — ink that reaches a tip fades out on the surface itself.
The only thing that changes is the colour of the surface inside the outline. Black Rorschach ink runs up the grey hand — ragged blots drawn off the wrist and across the back of the hand, splitting around the knuckles into separate streams that are pulled out along each finger, narrowing and stretching as they travel toward the tips and thinning away to bare grey at the ends, while fresh ink keeps welling up at the wrist behind them.
```

## 3. plane_waves — Hokusai swell running down the fuselage

THIS is the figure's fourth panel. The asset is `data/plane_waves` (mesh
`plane_waves_render_frame.obj`, video `plane_waves.mp4`), run
`runs/rung27_l1_lp_mcfmv2_D_all_qkvo+sa_r4_s42_6651791a`. It sits outside the paper's 24
(batches A+B+C) — `3DV_SUBMISSION_FIGURE_LIST.md:86` lists it as a bench asset to promote
if a slot opens. Its training-view render predates jobs/fixview_arm.sbatch and lives under
the older `out/view_plane_waves_mcfm_train` tag, same camera, same composite.

Start frame is a deep-navy plane carrying cream Hokusai wave crests with clawed foam tips
and fine engraved flow lines. Two things make this prompt different from the other three:
the style is a **flat woodblock print**, which a video model will happily drift into
photoreal ocean water, and woodblock prints carry signature cartouches, which is why the
no-writing lock matters far more here than anywhere else.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THIS IS A WOODBLOCK PRINT IN MOTION: the surface of the plane is a Japanese ukiyo-e wave print that has come alive. The water is moving across it continuously for the entire clip. Every wave is travelling at every moment. Nothing is ever still, ever frozen, ever holding a pose.
THE SWELL RUNS DOWN THE BODY AND KEEPS RUNNING: the waves travel steadily along the plane from the nose, back down the fuselage, and outward along both wings to their tips — all of them marching the same way, none of them stationary. At least eight separate cream crests pass any given point on the fuselage over the course of the clip. A crest runs the length of the fuselage in about two seconds. This is fast, obvious, unmistakable travel.
EVERY CREST CROSSES THE WHOLE BODY: no wave sits in one place and swells. Each cream crest rises at the nose, runs the full length of the plane, breaks outward along a wing, and thins away at the wingtip, and a new one is already forming behind it. At least five distinct crests are visible somewhere on the plane in every single frame.
THE CLAWED FOAM REACHES AND CURLS: the tip of every crest breaks into the sharp finger-like claws of a Hokusai wave. Those claws extend forward, curl over, splay apart into smaller claws, and collapse back into the navy as the crest passes, then form again on the next crest. The claws are never static decoration — they are always reaching, always curling, always dissolving.
THE ENGRAVED LINES STREAM: the fine parallel carved lines inside the navy water flow along with the swell, bending around the crests and streaming back down the fuselage and out along the wings, so even the dark areas are visibly in motion and never read as flat empty paint.
ONLY TWO COLOURS, ALWAYS FLAT: the water stays exactly two flat inks — deep indigo navy and pale cream — with hard carved edges between them. It never becomes photographic water, never gains transparency, depth, reflections, sparkle, spray or realistic lighting. It stays a printed woodblock surface from the first frame to the last.
NO WRITING OF ANY KIND: there are no letters, kanji, hiragana, katakana, glyphs, symbols, numerals, seals, signature cartouches, red stamps, inscriptions or logos anywhere on the plane or in the frame. Only waves.
THE PLANE ITSELF NEVER MOVES: the plane is a single rigid object — no banking, pitching, rolling, yawing, no flying, no vibration, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. Only the water on it moves; the aircraft underneath is bolted down.
THE LIGHTING DOES NOT TRAVEL: the plane's shading and the soft shadow under the fuselage and wings are painted onto the model and stay exactly where they are in the input image. The waves slide over them. The image as a whole never slides, and the plane must never look like it is turning or flying.
THE FORM MUST STAY READABLE: the nose, the swept wings, the tail fin, the tailplane and the trailing edges stay clearly visible and separable in every frame, and the plane is never flattened into a solid navy silhouette.
THE BACKGROUND NEVER CHANGES: every pixel outside the plane's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE WATER NEVER LEAVES THE OUTLINE: everything is on the surface, flat against it, contained entirely inside the outline. No spray, no droplets, no splash, no foam claw crosses the edge or extends past it into the background, and nothing is emitted into the air.
The only thing that changes is the colour of the surface inside the outline. Hokusai swell rolls down the plane — cream crests rising at the nose and running back along the fuselage, breaking outward across each wing, their clawed white tips reaching forward and curling over before collapsing into the indigo, carved parallel lines streaming along beneath them, wave after wave arriving in an unbroken procession that never pauses and never repeats.
```

## 4. chair_real_wooden_crack — peel wave running up the chair

Best start frame of the five: teal paint against orange-tan wood at roughly 50/50, with
crisp dark outline strokes around every peel shape. That balance is what the conservation
lock needs — plenty of room to travel in both directions.

Peeling paint is the one effect here whose natural physics is *one-way and terminal*:
paint comes off and does not go back on. A straight "the peel advances" prompt therefore
ends with a bare chair and no texture left to move. The fix is a **rolling peel wave** —
the paint is barely bonded, lifts at the leading edge, and lays back down behind — which
gives direction and trackable features while conserving coverage by construction. The
extra lock this object needs and the others do not is **nothing falls off**: flakes
dropping is the most natural thing for peeling paint to do, and it would break the
silhouette immediately.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE PAINT IS LIFTING IN A ROLLING WAVE: the blue paint on this chair is barely bonded to the wood. It lifts off in a wave that rolls continuously along every part of the frame for the whole clip. Every painted area is in motion at every moment. Nothing is ever still, ever frozen, ever holding a pose.
THE PEEL WAVE RUNS UP THE CHAIR AND KEEPS RUNNING: the wave travels steadily upward — from the feet, up each leg, across the seat, out along the arms and up the back rest — all of it moving the same way, none of it stationary. At least six bands of bare wood pass any given point on a leg over the course of the clip. A band runs the full length of a leg in about two seconds. This is fast, obvious, unmistakable travel.
EVERY BAND CROSSES ITS WHOLE MEMBER: no patch of bare wood sits in one place and grows. Each band of exposed wood opens at one end of a leg, rail, arm or back slat, sweeps along the entire length of that member, and closes again at the far end, and the next band is already opening behind it. At least five separate bands are visible somewhere on the chair in every single frame.
THE CURLED EDGE IS THE THING TO WATCH: at the leading edge of every band the blue paint curls up off the wood in a thin ragged lip outlined in dark brown, splits into fingers, and peels back. At the trailing edge the paint settles flat onto the wood again and the dark outline closes behind it. Those curled lips are sharp and trackable and they travel along the timber; they never sit in one spot flickering.
PAINT AND WOOD TRADE PLACES AS THE WAVE PASSES: exactly as much of the chair is blue and exactly as much is bare wood at the end as at the beginning. Wherever a band opens to wood, the band behind it closes back to paint at the same moment, because it is one wave moving, not paint disappearing. The chair never strips to bare wood and never seals to solid blue. Nothing accumulates.
THE GRAIN RUNS WITH THE WOOD: the orange-tan wood revealed under each band carries fine dark grain lines running along the length of the member, and those lines stay fixed to the wood as the bands sweep over them — the grain belongs to the chair, not to the wave.
TWO COLOURS AND A DARK OUTLINE: the surface stays flat teal-blue paint and flat orange-tan wood, separated by hand-drawn dark brown outline strokes exactly like the input image. It never becomes photographic, never gains gloss, wetness, depth, sparkle or realistic lighting, and no third colour appears.
NOTHING FALLS OFF: no flake, chip, curl, splinter, sliver or fragment ever detaches, drops, falls, flutters or leaves the chair. The lifted paint stays attached at its base, curls in place, and lays back down. There is no debris anywhere, on the chair or below it.
THE CHAIR ITSELF NEVER MOVES: the chair is a single rigid object — no rocking, wobbling, tipping, creaking, settling, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. Only the paint on it moves; the chair underneath is bolted down.
THE LIGHTING DOES NOT TRAVEL: the chair's shading, the shadows in the joints and under the seat, and its highlights are painted onto the model and stay exactly where they are in the input image. The peel wave slides over them. The lit faces stay lit and the shadowed faces stay shadowed as bands pass through. The image as a whole never slides, and the chair must never look like it is turning.
THE FORM MUST STAY READABLE: the four legs, the seat, the two arms, the arm supports and the slats of the back rest stay clearly visible and separable in every frame, and the chair is never flattened into a solid blue or solid brown silhouette.
THE BACKGROUND NEVER CHANGES: every pixel outside the chair's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE EFFECT NEVER LEAVES THE OUTLINE: everything is on the surface, flat against it, contained entirely inside the outline. Nothing crosses the outline, nothing extends past the edge, nothing drips, flakes, spills or radiates into the background, and nothing is emitted into the air.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the chair. Only paint and wood.
The only thing that changes is the colour of the surface inside the outline. A wave of peeling paint rolls up the chair — teal paint curling off the wood in ragged dark-outlined lips that sweep up each leg, across the seat and up the back slats, bare orange-tan wood opening behind each lifting edge and closing back to blue as the wave moves on, band after band travelling up the frame in an unbroken procession that never pauses and never repeats.
```

## Appendix. napolean_waves — WRONG MESH, kept for reference

Written against the Napoleon bust by mistake when the figure panel is the PLANE.
Valid for that object if it is ever wanted, but it is not the figure asset. The v3 fix
it describes applies to every prompt here and should be back-ported to the other two: the earlier
drafts were roughly 80% negation with a single drive clause buried in the last line. That
buys obedience and no motion. Here the motion blocks are FRONT-LOADED ahead of the
restrictions, the effect is framed as real ocean physics rather than as an abstract
"pattern that travels", and the travel is given a **countable rate** — crests per clip —
which is something the model can be held to in a way that "a real distance" is not.

Start frame is the bust fully covered in dark blue water with a white foaming region across
the chest. Path: swell runs horizontally across the body and up over the skull.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THIS IS A LIVE OCEAN, NOT A PICTURE OF ONE: the surface of the bust is open sea in motion. Deep-water swell rolls across it continuously for the entire clip. Every part of the water is moving at every moment. Nothing about the water is ever still, ever frozen, ever holding a pose.
THE SWELL RUNS ONE WAY AND KEEPS RUNNING: the waves travel steadily across the bust from the far shoulder, over the chest, past the near shoulder, and up across the jaw and skull — all of them marching the same way, none of them stationary. At least eight separate wave crests pass any given point on the chest over the course of the clip. A crest crosses from one shoulder to the other in about two seconds. This is fast, obvious, unmistakable travel.
EVERY CREST CROSSES THE WHOLE BODY: no wave sits in one place and swells. Each crest rises at one edge of the surface, runs the full width of the bust, and disappears off the far edge, and a new one is already entering behind it. At least five distinct crests are visible somewhere on the bust in every single frame.
THE FOAM IS BORN, STREAKS, AND DIES: white foam forms on the front face of each advancing crest, gets dragged backward into long streaks down the wave's back as the crest overtakes it, thins into lacy filaments, and dissolves into the dark blue before the next crest arrives. Foam never sits still and never simply pulses brighter and dimmer in place.
THE WATER IS DEEP BLUE BETWEEN THE CRESTS: the troughs are dark navy and near-black in the hollows, so each pale crest reads clearly against them and can be followed with the eye across the whole body from one side to the other.
THE BUST ITSELF NEVER MOVES: the bust is a single rigid sculpture — no head turn, no tilting, no breathing, no rotation, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. Only the water on it moves; the sculpture underneath is bolted down.
THE LIGHTING DOES NOT TRAVEL: the bust's sculpted shading, the shadow under the jaw and along the far side of the head, and its specular highlights are painted onto the stone and stay exactly where they are in the input image. The waves slide over them. The lit side stays lit and the shadowed side stays shadowed as crests pass through. The image as a whole never slides, and the bust must never look like it is turning.
THE FACE MUST STAY READABLE: the brow, the closed eyes, the nose, the lips, the jawline, the ear and the collar of the coat stay clearly visible in every frame and are never washed out into flat blue.
THE BACKGROUND NEVER CHANGES: every pixel outside the bust's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE WATER NEVER LEAVES THE OUTLINE: everything is on the surface, flat against it, contained entirely inside the outline. No spray, no droplets, no splash, no mist, no wave crosses the edge or extends past it into the background, and nothing is emitted into the air.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the bust. Only water.
The only thing that changes is the colour of the surface inside the outline. Ocean swell rolls across the bust — dark navy crests rising at one shoulder and marching steadily across the chest, white foam tearing off each crest and streaking backward down its face before dissolving, the deep troughs between them sliding along behind, wave after wave arriving in an unbroken procession that never pauses and never repeats.
```

## 5. bob — spots gliding over the whole duck  (v3)

Conditioning image `supplementary_meshes/front/bob_yaw250.png` — up=+Y, yaw 250, pitch 20,
28.6% exposed, both eyes visible.

**Three versions, and what each got wrong.**
v1 asked for an orbit with a frequency ("six spots past the tail, three seconds tail to
head") and came back as a conveyor belt. v2 replaced the rate with "slowly" and stacked
*gentle · unhurried · never a march · never a scroll · never a rush* — four anti-speed
clauses and no floor — and Kling resolved the pile the cheapest way available: the spots
stopped moving.

v3 gives speed **a floor and a ceiling**, both as distance in units the model can see
(spot widths), and adds an explicit failure test for the frozen case. The rule that came
out of this: a bound stated only as "not fast" collapses to zero. Every speed spec in this
file needs both ends.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE SPOTS ARE ALWAYS MOVING: the coloured spots slide steadily across the duck's surface for every second of the clip. This is the subject of the shot. There is no settling, no pausing, no holding — from the first frame to the last, every spot is in motion.
NEVER STATIONARY: if you compared any two frames one second apart, every spot would be in a visibly different place. A spot that stays put for even a moment is a failure. The pattern is never allowed to look like a still image.
HOW FAR EACH SPOT GOES: over the clip each spot travels at least three times its own width, and at most about a quarter of the way around the duck. Three spot-widths is the minimum — less than that reads as frozen. A quarter turn is the maximum — more than that reads as spinning.
THEY ALL DRIFT THE SAME WAY: every spot moves in the same direction across the surface — up along the body, forward along the neck, over the top of the head, and down past the cheek — carried together like markings on one current. None drifts backwards and none wanders off on its own.
THE SPOTS CROSS THE HEAD TOO: the head and neck are part of the surface the spots travel over, not a zone they avoid. Spots glide up out of the body onto the neck, across the top and side of the head, and continue down the far side.
THE COUNT NEVER CHANGES: no spot is created and none vanishes. The same spots keep drifting for the whole clip. The duck never becomes more spotted or less spotted, the yellow never fills in, and the body never turns solid blue.
EACH SPOT KEEPS ITS OWN IDENTITY: every spot holds its own size and colour — big mid-blue ones stay big and mid-blue, small navy ones stay small and navy, pale white ones stay pale — so one spot can be picked out in the first frame and followed to the last. They stretch a little riding over a curve and settle again, but never dissolve into noise, never blink out, never teleport, never trade colours.
THE FACE STAYS PUT: both black eyes and the orange beak are moulded features, not markings in the drift. They stay exactly where they are and exactly the colour they are for the whole clip. A spot that reaches an eye glides around it and continues past; it never covers an eye, merges with one, or drags one along, and no extra eye ever appears.
THE HOLE STAYS EMPTY: the ring's hole is background, not surface. Every pixel inside it is the same solid uniform white as the background, in every frame. Nothing grows across it, no spot enters it, no film or web ever forms there, and it never closes or shrinks.
THE DUCK ITSELF NEVER MOVES: the duck is a single rigid moulded object — no bobbing, rocking, floating, turning, tilting, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. Only the spots move; the duck underneath is bolted down.
THE LIGHTING DOES NOT TRAVEL: the duck's shading, the soft shadow under the body and inside the ring, and the glossy highlight along the top are properties of the moulded plastic and stay exactly where they are in the input image. The spots slide underneath them. The lit top stays lit and the shaded underside stays shaded as spots pass through. The image as a whole never slides, and the duck must never look like it is rotating.
THE FORM MUST STAY READABLE: the head, the beak, both eyes, the curve of the body and the opening of the ring stay clearly visible in every frame, and the duck is never flattened into a solid silhouette.
THE BACKGROUND NEVER CHANGES: every pixel outside the duck's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
THE SPOTS NEVER LEAVE THE OUTLINE: everything is on the surface, flat against it, contained entirely inside the duck's outline. No spot crosses the edge, drips, peels, floats off, or radiates into the background.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the duck. Only spots.
The only thing that changes is the colour of the surface inside the outline. Blue and white polka dots slide across the yellow rubber duck — each spot travelling visibly along the curve of the body, swelling as it crosses the widest part and narrowing as it turns away, climbing onto the neck and passing over the head before easing down the far side, the whole field of spots carried together in one continuous current that never once comes to rest.
```

## 6. blub — iridescent bands travelling head to tail

Keenan Crane's `blub`, a blue fish: blue flanks, cream-to-orange dorsal fin and tail, two
black eyes, pale lavender belly and mouth. Conditioning image
`supplementary_meshes/front/blub_yaw030.png` — up=+Y, yaw 30, pitch 20, 26.4% exposed,
rendered through the Kling camera with blub's own `blub_texture.png`.

**Why yaw 30 and not the 34.5% peak.** yaw 90/270 expose the most surface but are pure side
profiles showing one eye, and a feature visible from exactly one side is unsupervised — the
adapter can put anything on the far one. yaw 30 is the 3/4 view: both eyes, mouth, dorsal
fin, tail and flank all in the loss, for 8 points of exposed area. Same trade as bob at
yaw 250.

Blub is genus 0, so no hole lock. Its own three locks are the **two eyes**, the
**cream fins** (which will otherwise go blue as the bands sweep over them) and the **pale
belly**, all fixed markings on a body whose whole point is that colour moves across it.

```
Static tripod framing; the camera is locked — no pan, tilt, roll, dolly, push-in, pull-out, zoom, reframing, or shake.
THE BANDS DRIFT, SLOWLY: broad soft bands of colour move continuously along the fish's flank, the way iridescence slides over real fish scales. The motion never stops, but it is gentle and unhurried throughout — a slow drift, never a march, never a scroll, never a rush, never a flicker.
HOW FAR THEY GO: over the entire clip each band travels only about one head-length along the body. A viewer should be able to pick one band behind the eye and watch it ease back toward the tail without it ever reaching the tail fin. If a band crosses the whole fish, it is moving far too fast.
THEY ALL DRIFT THE SAME WAY: every band moves from the head toward the tail, all of them carried together like markings on one slow current. None sits still, none drifts forward, none wanders off on its own.
THE BANDS CROSS THE WHOLE FLANK: the bands sweep across the cheek, the gill area, the belly and the deep blue of the flank, curving to follow the roundness of the body. No part of the blue is ever left as flat unchanging colour.
NEW BANDS FORM AT THE HEAD: as a band eases away toward the tail and fades out, a new one forms near the head behind it, so there are always three or four bands somewhere on the fish. The fish never turns uniformly pale and never turns uniformly dark. Nothing accumulates and nothing drains away.
THE COLOUR STAYS IN FAMILY: the bands shift the blue between deeper navy, mid cerulean and a lighter cyan sheen. They never introduce red, green, purple or yellow into the flank, and the fish never stops reading as a blue fish.
THE EYES STAY PUT: both black eyes, each with its small white highlight, are fixed features of the fish. They stay exactly where they are, exactly the size and colour they are, for the whole clip. Bands pass behind and around them without covering them, dragging them, smearing them or dimming them, and no third eye ever appears.
THE FINS AND BELLY KEEP THEIR COLOURS: the dorsal fin and tail stay their cream-to-orange, and the pale lavender belly and mouth stay pale. The blue bands stop at these boundaries — they never wash over the fins, never tint them blue, and never spread across the belly patch.
THE FISH ITSELF NEVER MOVES: the fish is a single rigid object — no swimming, no fin flick, no tail sweep, no breathing, no turning, no deformation, no micro-jitter. Its outline, scale and position in the frame are pixel-identical to the input image in every frame. Only the colour on it moves; the fish underneath is bolted down.
THE LIGHTING DOES NOT TRAVEL: the fish's shading, the shadow under the belly and along the far flank, and the soft highlight on the top of the back are properties of the moulded surface and stay exactly where they are in the input image. The bands slide underneath them. The lit side stays lit and the shaded side stays shaded as bands pass through. The image as a whole never slides, and the fish must never look like it is turning.
THE FORM MUST STAY READABLE: both eyes, the mouth, the dorsal fin, the pectoral fin and the tail stay clearly visible in every frame, and the fish is never flattened into a solid silhouette.
THE BACKGROUND NEVER CHANGES: every pixel outside the fish's outline is the same solid uniform white as the input image, in every single frame from first to last — flat, empty, untouched, with no gradient, no shadow, no tint, no light.
NOTHING LEAVES THE OUTLINE: everything is on the surface, flat against it, contained entirely inside the fish's outline. No band, sheen, bubble or glow crosses the edge or radiates into the background, and nothing is emitted into the air.
NO WRITING OF ANY KIND: there are no letters, glyphs, symbols, characters, numerals, inscriptions or logos anywhere on the fish. Only colour.
The only thing that changes is the colour of the surface inside the outline. Slow iridescent bands travel down the blue fish — soft edges of lighter cyan easing back from the cheek along the flank, deepening to navy behind them, curving with the roundness of the body and fading out before the tail as a new band forms up near the head, the whole field of colour carried together in one slow unhurried current that never stops and never repeats.
```
