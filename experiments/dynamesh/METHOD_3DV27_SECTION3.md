Given a driving video of a surface effect, a clip in which
the camera is fixed, the object does not move, and only its
appearance evolves, together with a 3D mesh, our goal is
to produce a texture on that mesh that follows the video
over time without flickering. We propose DynaMesh, which
conditions a frozen image-to-3D generator on a window of
video frames rather than on one frame at a time, and adapts
a small part of the model per scene to recover the detail the
frozen generator loses.

The driving video provides frames I1, . . . , IT . The generator, TRELLIS.2 [38, 39] in our implementation, separates shape from appearance. A shape encoder maps the
input mesh to a sparse structure S, a set of occupied voxels each carrying a latent feature. A flow model predicts
per-voxel appearance features conditioned on S and on an
input image, encoded into tokens by a pretrained feature
extractor, and a decoder expands those features into surface
color for the mesh that S encodes. We run this generator
once per video frame, and modify its use in three ways: the
conditioning tokens are blended over a temporal window
before the generator attends to them (Sec. 3.1), low-rank
adapters fit to the driving video restore detail (Sec. 3.2), and
the mesh, the sparse structure S, and the denoising noise are
shared across all frames, so that every change over time is
carried by appearance alone (Sec. 3.3).

The image-to-3D generator stays frozen throughout. All
image-space supervision is restricted by silhouette matching between pixels of the driving frame and our render, so
the losses compare texture and never shape.

## 3.1. Multi-Frame Conditioning

A frozen image-to-3D generator conditioned on a single
frame commits to that frame alone. Run independently
down a video, every frame is decided in isolation, and the
resulting textures flicker: details appear, vanish, and reappear from one frame to the next even when consecutive
inputs are nearly identical. As discussed in Sec. 2, video
synthesis faces the same failure, and its remedies share one
principle: the generation of a frame should be informed by
its temporal neighborhood. We transfer this principle into
the conditioning of the 3D generator.

Let z_t denote the conditioning tokens of frame I_t, computed once by the frozen feature extractor. Before the generator's cross-attention sees them, we replace each token by
a blend over a temporal window W centered at t,

  z̃_t^i = Σ_{δ∈W} α_δ^i z_{t+δ}^i,   α^i = softmax_{δ∈W} ( ⟨z_t^i, z_{t+δ}^i⟩ / √D )     (1)

where i indexes token positions and D is the token dimension. Each position attends over the same position in the
neighboring frames, with the current frame as the query.
Where consecutive frames agree, the blend averages them
and suppresses the frame-to-frame noise; where the effect
genuinely changes, the current frame dominates its own
weight and the change passes through. The mechanism is
parameter-free, since the weights come from token similarity alone, so the generator itself is untouched while generation at frame t is anchored to its neighborhood in time.

We use the three-frame window [t−1, t, t+1] by default,
and evaluate a two-frame window [t, t+1] as well as a joint
variant, in which a token may attend to any position in any
frame of the window, as ablations in Sec. 4.

Sharing the denoising noise across frames, the anchor
common in video synthesis [24], already removes a large
share of the flicker on its own, so we do not claim temporal conditioning as the sole anti-flicker mechanism. Section 4 separates the two contributions, reporting neither,
fixed noise only, conditioning only, and both, and shows
what conditioning adds once the noise is already fixed.

## 3.2. Per-Scene Adaptation

Temporal conditioning stabilizes the sequence, but the
frozen generator still loses detail relative to the driving
video: the effect washes out, arriving as diffuse patches
where the video shows sharp structure. We recover this
detail with low-rank adapters [20] fit per scene against the
driving video, while everything else stays frozen. Writing
W for a frozen weight matrix, the adapted layer computes

  W′x = Wx + s·BAx,     (2)

with a low-rank residual BA and a scale s. Across all
adapted layers, the residuals amount to less than a tenth of
a percent of the generator's parameters.

The generator offers two distinct places for this capacity, and they play different roles. Adapting the projections
of the cross-attention, where the conditioning tokens enter, changes what the image is allowed to say. Adapting
the self-attention, which propagates information between
voxels and never reads the image, changes how appearance spreads across the object. We adapt both, attaching
rank-4 residuals to the query, key-value, and output projections of the cross-attention and to the self-attention projections of every generator block, and find in Sec. 4 that the
choice of module matters far more than the choice of projection within a module. Self-attention placements require
care. Supervised from a single camera, an adapter that reshapes propagation between voxels can raise the training-view score by routing texture toward the supervising camera
rather than binding it to the surface, and no training-view
metric detects the difference. We therefore judge adapter
placements by view consistency under orbit renders, where
the camera passes every surface point at more than one moment of the evolution so that a texture living on the surface
must agree with itself, alongside reconstruction of the driving view.

The video frame itself cannot serve as the training target, because the video model reshapes the object over the
clip: its silhouette grows and drifts while our render's silhouette is pinned by the fixed mesh and camera. A pixel
loss against the raw frame would charge the adapter for that
shape difference as though it were texture error. We therefore build, for every frame, a target that carries our mesh's
outline. Since neither camera nor object moves, each pixel
inside our silhouette looks at the same surface point in both
images, so we copy the video's color at exactly the pixels
our render occupies and leave the rest background. The loss
is an L1 penalty between render and target with a small perceptual term, computed inside the silhouette. Each training
step runs the generator to a random point along its denoising trajectory with gradients disabled, takes a single step
with gradients enabled, decodes and renders the result, and
updates the adapters only.

## 3.3. Stabilizing Geometry Across Frames

Run per frame with a generated shape, an image-to-3D generator returns a different mesh every frame, with different
vertices and a different triangulation, even when consecutive renders look alike. This is invisible in image space
and fatal for the output: a per-frame texture attached to a
per-frame mesh is not an asset that downstream tools can
animate, relight, or edit.

Our pipeline removes this by construction. The mesh is
an input, and we encode it once: the shape encoder maps
it deterministically to the sparse structure S, the only place
geometry enters the generator, and the same S conditions
the appearance flow at every frame. With the denoising
noise also shared across frames (Sec. 3.1), everything the
generator computes about shape is identical from frame to
frame, and the per-frame conditioning tokens can influence
appearance alone. The output of the full pipeline is a single
mesh with one triangulation and a texture per frame, one
object whose surface changes over time.
