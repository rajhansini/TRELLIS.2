# Results — temporal metrics (draft text for the paper)

## Why texel-space, and not pixels

Both temporal metrics are computed on the **PBR voxel field** the flow model
predicts, not on rendered frames. Two consequences, and both are the reason the
numbers mean anything:

1. **They are view-independent.** A pixel-space temporal metric confounds texture
   change with camera motion, so it can only be read at a fixed camera. A texel
   metric is defined on the object, so a single number covers the whole surface,
   including parts no camera in the evaluation ever saw.
2. **They see the quantity the method actually produces.** Rendering is a lossy
   projection; shading, occlusion and rasterisation all add their own frame-to-frame
   variation. Measuring before the renderer removes that confound.

A prerequisite is that consecutive frames describe the *same* voxels. The
implementation asserts this per frame (`torch.equal` on the coordinate tensor);
without it a difference between frames could be comparing different points in space.

## The two metrics

Write $c_t \in \mathbb{R}^{V \times 3}$ for the base-colour channel of the predicted
field at frame $t$, over $V$ occupied voxels.

**Texel flicker** — the mean first temporal difference:
$$\mathrm{Flicker} = \frac{1}{T-1}\sum_{t=2}^{T} \big\| c_t - c_{t-1} \big\|_1$$
How much the surface changes from one frame to the next. It is a *rate*: a texture
that evolves quickly has high flicker whether or not it is smooth, so flicker alone
cannot be read as a quality score.

**Texel acceleration** — the mean second temporal difference:
$$\mathrm{Accel} = \frac{1}{T-2}\sum_{t=3}^{T} \big\| c_t - 2c_{t-1} + c_{t-2} \big\|_1$$
How abruptly that rate itself changes. This is the quantity that corresponds to what
a viewer calls flickering: a texture evolving fast but *steadily* has high flicker and
low acceleration, whereas one that jitters back and forth has high acceleration even
if its net motion is small. Acceleration is therefore the metric that separates
temporal coherence from temporal speed.

> **Naming.** This quantity was called *texel jerk* in earlier internal results. That
> is a misnomer worth correcting before submission: in kinematics *jerk* is the
> **third** derivative, while this is the **second**. *Texel acceleration* is the
> accurate term; *temporal roughness* is an acceptable plainer alternative. Any
> reviewer with a physics background will flag "jerk".

**Texel drift** — first frame against last, $\|c_T - c_1\|_1$ — is reported as a
**guard, not a score**. Flicker and acceleration are both minimised by a degenerate
solution: a texture that does not move at all. Drift detects exactly that failure. It
is only meaningful read alongside the other two, and we do not mark a "best" value
for it.

## What the numbers say

Averaged over 15 objects, adding temporal context lowers flicker by 15.6% and
acceleration by 33.5%, while PSNR and SSIM each improve by about 1%. Drift is
unchanged (+0.0%), so the coherence gain is not bought by slowing the texture down —
which is the specific artefact the guard exists to rule out.

PSNR cannot settle this question on its own. It scores each frame against its own
target independently and is therefore structurally blind to temporal coherence: two
sequences with identical per-frame error, one smooth and one alternating, receive the
same PSNR. That is why the temporal metrics carry the claim and PSNR is reported
alongside only to show fidelity is not sacrificed.
