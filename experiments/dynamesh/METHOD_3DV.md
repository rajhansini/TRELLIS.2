# Low-rank adaptation of cross-attention for conditional texture synthesis

**Method section draft — 3DV submission**

We adapt a frozen 3D generative model to a new texture-conditioning objective by
training a low-rank residual on the cross-attention key/value projection of every
block — 0.038% of the parameters of the generator being adapted, and 0.012% of the
full pipeline. The placement is determined by the architecture rather than chosen
empirically: shape and image reach the generator through structurally distinct
pathways, and only one of them is the pathway we intend to modify.

---

## 3.1 Backbone

We build on TRELLIS.2, which synthesises texture for a supplied mesh through a
rectified-flow transformer over a sparse voxel latent. The generator is a
1.3B-parameter DiT of **30 blocks**, width 1536, 12 heads, with rotary position
embedding and shared adaptive-LayerNorm modulation. Texture is produced by
integrating the learned velocity field over **12 Euler steps**.

Three components matter for what follows. A shape encoder maps the input mesh to a
sparse latent $S \in \mathbb{R}^{N \times 32}$ over occupied voxels. A DINOv3
ViT-L/16 encoder maps the conditioning image to a token sequence
$C \in \mathbb{R}^{1029 \times 1024}$. A decoder expands the generated texture
latent to per-voxel appearance, which is rasterised through the fixed mesh to form
an image.

Geometry is never predicted. The generator consumes the mesh and emits appearance
channels only, none of which is a position, so the method modifies texture and
nothing else *by construction* rather than by regularisation.

---

## 3.2 Where to adapt, and why

The generator is conditioned on two signals, and they do not enter the same way.

**Shape enters by concatenation.** The shape latent is concatenated channel-wise
with the noise latent at the input layer, giving the 64 input channels the model
expects, and is thereafter indistinguishable from the state being denoised:

$$h^{0} = \mathrm{InputLayer}\big([\,z \,;\, S\,]\big), \qquad z \in \mathbb{R}^{N \times 32},\; S \in \mathbb{R}^{N \times 32} \tag{1}$$

**The image enters by cross-attention**, separately, in every one of the 30 blocks.
Each block applies self-attention, then cross-attention against the image tokens,
then an MLP:

$$
\begin{aligned}
x &\leftarrow x + \mathrm{gate}\cdot\mathrm{SelfAttn}\big(\mathrm{norm}_1(x)\big)\\
x &\leftarrow x + \mathrm{CrossAttn}\big(\mathrm{norm}_2(x),\, C\big)\\
x &\leftarrow x + \mathrm{gate}\cdot\mathrm{MLP}\big(\mathrm{norm}_3(x)\big)
\end{aligned}
\tag{2}
$$

This separation is the method's premise. **Self-attention is the only operator that
moves information between voxels** — it is what carries an edit to regions the
conditioning image does not depict. Cross-attention is the only operator through
which image evidence enters at all.

> **Placement principle.** Adapting cross-attention changes *what the image is
> allowed to say* about the texture. Leaving self-attention frozen preserves *how
> the model distributes what it is told* — a property learned from large-scale 3D
> data that a per-instance adapter has no basis to relearn. The adapter therefore
> specialises the conditioning pathway while inheriting the backbone's spatial
> reasoning intact.

The alternative placements fail this test. An adapter in the decoder sits
downstream of every cross-attention layer, so image evidence never reaches it and
the model's 3D reasoning has already concluded; it can only rescale the appearance
it is handed. An adapter on self-attention would modify the propagation mechanism
itself, which is the component we most want to preserve.

```
   noise z (N×32) ─┐
                   ├─► concat ─► ┌───────── × 30 BLOCKS (FROZEN) ─────────┐
   shape S (N×32) ─┘   (→64)     │  self-attn ─► cross-attn ─► MLP        │ ─► decoder ─► raster
                                 │  voxel↔voxel   to_kv + ΔW              │              (fixed mesh)
                                 └──────────────────▲────────────────────┘
                                                    │
                                     DINOv3 tokens C (1029 × 1024)
```

*Two conditioning pathways. Shape is fused once, by concatenation, before the first
block. The image is injected repeatedly, by cross-attention, inside every block. The
adapter is applied only to the cross-attention key/value projection, so the shape
pathway and the voxel-to-voxel self-attention are both left exactly as pretrained.*

---

## 3.3 Low-rank residual on the key/value projection

Cross-attention in each block carries three linear maps: a query projection from the
voxel state, a fused key/value projection from the image tokens, and an output
projection. We adapt the key/value map, the one that reads the image.

| Projection | Reads             | Shape        | Adapted  |
|------------|-------------------|--------------|----------|
| `to_q`     | voxel state       | 1536 → 1536  | —        |
| `to_kv`    | image tokens      | 1024 → 3072  | **yes**  |
| `to_out`   | attention output  | 1536 → 1536  | optional |

For a frozen weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ we learn a
rank-$r$ residual, leaving $W$ untouched:

$$W' x \;=\; W x \;+\; \frac{\alpha}{r}\, B A x \tag{3}$$

where $A \in \mathbb{R}^{r \times d_\text{in}}$ is initialised from a Kaiming-uniform
distribution and $B \in \mathbb{R}^{d_\text{out} \times r}$ is initialised to zero,
so $W' = W$ exactly at initialisation and training begins from the unmodified
generative prior.

The factor $\alpha/r$ is not cosmetic. Since $BA$ is a sum of $r$ rank-one terms,
its magnitude grows with $r$; without normalisation a rank sweep varies the
effective step size alongside the rank, and the two cannot be separated. Holding
$\alpha$ fixed makes rank an independent variable.

Applied to `to_kv` across all 30 blocks, the adapter costs
$r\,(1024 + 3072)\cdot 30 = 122{,}880\,r$ parameters — **491,520 at rank 4**, or
0.038% of the 1.3B generator. The residual is injected by forward hooks on the
projection modules, so the pretrained weights are never written to and the base
model remains recoverable at any point.

---

## 3.4 Training through the sampler

The supervision is an image, but the adapter acts inside a velocity field integrated
over 12 Euler steps. Differentiating the full trajectory would require retaining
activations for all 12 evaluations of a 1.3B-parameter transformer, which is not
tractable.

We instead differentiate **one knot of the real sampling schedule**. Each training
step:

1. samples a knot $k \sim \mathcal{U}\{0,\dots,11\}$ from the deployed 12-step schedule;
2. integrates the first $k$ steps under `no_grad` *with the adapter active*, reaching
   the state $x_k$ the deployed sampler would actually occupy;
3. evaluates the velocity **once** with gradient enabled, giving $v = f_\theta(x_k, t_k, C)$;
4. converts that velocity to a clean-latent estimate in closed form;
5. decodes, rasterises, and compares against the target image.

$$\hat{x}_0 \;=\; (1-\sigma)\,x_k \;-\; \big(\sigma + (1-\sigma)\,t_k\big)\, v \tag{4}$$

with $\sigma = 10^{-5}$ the rectified-flow floor. Because $k$ is drawn uniformly,
every point on the trajectory is supervised in expectation, at the cost of a single
graded transformer evaluation per step.

> **Why the real schedule.** The prefix is integrated with the deployed sampler,
> with the adapter active, rather than from a noised sample of the target. The state
> at which the gradient is taken is therefore one the model genuinely reaches at
> inference under its own adapted dynamics — the adaptation is trained on its own
> consequences rather than on an idealised trajectory.

---

## 3.5 Confidence-weighted image supervision

### Constructing the target

The supervision signal is the ground-truth appearance backprojected onto the mesh
and *re-rendered through the same mesh and the same camera* as the model's own
output. This is a deliberate choice: it removes silhouette mismatch as a confound by
construction rather than compensating for it in the loss. Prediction and target then
differ only in colour, and the rendered silhouettes agree pixel-for-pixel — verified
before every run.

It matters because the alternative has a known failure mode. When prediction and
target silhouettes disagree, the model is asked on the disputed band to paint
surface colour onto background. It cannot move geometry — the mesh is an input — so
it can only satisfy that demand by driving colour towards saturation, which appears
as blown-out regions along boundaries. Matching the render removes the demand rather
than penalising its consequence.

### Weighting the observations

Within the silhouette, pixels are still not equally informative about the surface
beneath them. A face seen close to head-on is resolved across many pixels; the same
face at a grazing angle is compressed into a sliver, so its colour is unreliable and
its surface footprint large. Treating both as equally authoritative lets the least
trustworthy observations drive the adaptation.

We weight each pixel by the Unstructured Lumigraph blending field of Buehler et al.,
applied to a training loss rather than to view blending:

$$w_p \;=\; \begin{cases} \cos\theta_p \,/\, a_p & \text{if } \cos\theta_p \ge \tau \text{ and } p \in V\\[2pt] 0 & \text{otherwise} \end{cases} \tag{5}$$

$\theta_p$ is the angle between the surface normal and the view direction, and $a_p$
the surface footprint of the pixel, estimated from finite differences of surface
position. $V$ is the set of pixels whose neighbours are also on the surface: at a
silhouette boundary the finite difference straddles the edge and the footprint
estimate is meaningless, so those pixels are excluded rather than assigned a
spurious weight. Observations below $\tau$ are discarded outright rather than
down-weighted, since grazing pixels are numerous enough that a soft weight still
lets them claim a substantial share of the loss.

The objective is the weighted squared error over the supervised region:

$$\mathcal{L} \;=\; \frac{\sum_{p \in M} w_p \,\big\lVert \hat{I}_p - I_p \big\rVert^2}{3 \sum_{p \in M} w_p} \tag{6}$$

$M$ is the intersection of the rendered and target coverage. Since the two
silhouettes already agree, what this removes is not a boundary conflict but the
residual set of pixels whose triangle carried a vertex the backprojection never
resolved: those pixels lie inside the silhouette yet carry no ground-truth colour,
and scoring them would supervise the model against an arbitrary value. Normalising
by $\sum w_p$ makes the objective invariant to the overall scale of the weights, so
only their relative magnitudes matter.

The optimiser is constructed over the adapter parameters alone, so the backbone is
never updated. Optimisation uses AdamW without weight decay. Because the appearance
decoder is half-precision, the loss is scaled by a fixed constant before
backpropagation: without it, appearance gradients underflow to exactly zero before
reaching the adapter, which presents as a silently untrained model rather than as an
error.

---

**Configuration.** Backbone TRELLIS.2-4B; texture flow 1.3B DiT, 30 blocks, width
1536, 12 heads, RoPE. Conditioning: DINOv3 ViT-L/16 image tokens; shape latent
concatenated at input. Sampler: rectified flow, Euler, 12 steps. Adapter: rank-$r$
residual on cross-attention `to_kv`, all blocks.
