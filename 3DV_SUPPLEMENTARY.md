# 3DV Supplementary — Itai's list, with the artifact behind each item

One entry per item Itai asked for. Each carries the artifact to pull numbers or
diagrams from, so nothing has to be re-derived when the supplement is written.

**Status vocabulary.** `HAVE` = the artifact exists and its content was opened and
checked. `LISTED` = the artifact exists in the gallery and its title matches, but I
have not opened it, so treat the link as a pointer, not a verified source. `NONE` =
no artifact exists; the data may still be on disk, and that is stated per item.

Numbers in this file that are marked verified were recomputed from `runs/*/config.json`
plus `final_eval.json` and `out/TEXEL/*.json`, not read from any status document.

---

# Figures

## 1. Gallery — more meshes and effects, at least four

New objects/effects beyond the 42, as an additional gallery panel.

| status | artifact | what to pull |
|---|---|---|
| LISTED | [Bunny — Shine](https://claude.ai/code/artifact/dfe2c348-c3f0-494e-9cba-70d80eec81e9) | new-mesh result, shine |
| LISTED | [Doorknob — Spinodal](https://claude.ai/code/artifact/b79a7157-b402-4041-b622-a0892f47a123) | new-mesh result, spinodal |
| LISTED | [Doorknob — BZ Waves](https://claude.ai/code/artifact/2186ac8e-ce65-4ec7-822b-ef4238cedb40) | new-mesh result, BZ waves |
| LISTED | [Doorknob and Bunny Targets](https://claude.ai/code/artifact/2a8328b6-cf47-4cd6-b514-abffff2862ae) | the conditioning renders and targets for the two above |
| LISTED | [Every Object, Every Method](https://claude.ai/code/artifact/1fbd4f9c-8cfe-440f-8aa4-d133a5fecec4) | the pool the gallery draws from |
| LISTED | [Rung Ladder Dailies](https://claude.ai/code/artifact/42db3ff0-0cc5-4ce7-92e1-0a03fec3b293) | index across objects |

**Gap.** Itai separately asked for **rust on a car** and **rust on Dale's robot**.
Meshes are on the cluster and watertight (`ddecatur/latent-nerf/shapes/nascar.obj`,
`supplementary_meshes/OBJ/robot_dale.obj`), views solved, prompts written. Blocked on
Kling videos plus a 30-epoch fit each. No artifact yet.

## 2. Comparison — at least two more

| status | artifact | what to pull |
|---|---|---|
| LISTED | [42-Object Comparison](https://claude.ai/code/artifact/ec05e3f8-0476-441d-b314-e976ae846d06) | the published 42-object comparison on the 2D-copy reference |
| LISTED | [Method Comparison Gallery](https://claude.ai/code/artifact/6205c3c5-ea75-45a5-9ec9-866d1accc008) | side-by-side panels across methods |
| LISTED | [Ivysaur Petal — All Methods](https://claude.ai/code/artifact/b64e1f34-8d12-4220-a717-84231e81ef82) | per-object all-methods strip |
| LISTED | [Airplane Blub — All Methods](https://claude.ai/code/artifact/446f9231-3a5e-4f1c-91e3-3ad4afe53dc4) | per-object all-methods strip |
| LISTED | [Spot Lava — All Methods](https://claude.ai/code/artifact/71e96e9b-19cb-4fba-8f85-7446cf918d32) | per-object all-methods strip |

There is one "All Methods" page per object across the 42, so picking two more
comparison figures needs no new compute, only selection.

**Do not cite.** [Window 11 Comparison](https://claude.ai/code/artifact/3cb51284-bce8-4a76-991d-56f01a0166c6)
is built on the raw-frame reference and reads about 6 dB low on PSNR/SSIM.

## 3. Window size ablation — including W=13 and W=15, shown as a graph

| status | artifact | what to pull |
|---|---|---|
| **HAVE** | [Window Ablation](https://claude.ai/code/artifact/65086efd-f633-44e4-a564-35e8441890e7) | the seven-row table, the flicker/PSNR graph, and the `window_ablation.tex` block |

Both of Itai's bullets are covered on that one page. The graph is a single chart:
window size on x, flicker on the left axis, PSNR on the right. Verified 42/42 objects
at every width.

| W | PSNR | SSIM | Flicker | Accel. |
|---|---|---|---|---|
| 1 (no blend) | 24.80 | 0.7141 | 0.00707 | 0.00971 |
| 3 (ours) | 24.84 | 0.7149 | 0.00577 | 0.00593 |
| 5 | 24.86 | 0.7160 | 0.00543 | 0.00516 |
| 7 | 24.85 | 0.7157 | 0.00527 | 0.00496 |
| 11 | 24.82 | 0.7149 | 0.00506 | 0.00474 |
| 13 | 24.81 | 0.7143 | 0.00498 | 0.00468 |
| 15 | 24.79 | 0.7140 | 0.00494 | 0.00466 |

Files: `experiments/dynamesh/out/window_sweep.{json,tex,pdf,png}`,
built by `jobs/build_window_figure.py`.

Related, a different operator, do not conflate: [Temporal Configuration](https://claude.ai/code/artifact/a71a121a-0dc5-4b39-bd11-4c0e6d943cdb)
crosses window width with blend flavour.

## 4. Existing textures — one or two more examples

Fig. 7 in the paper is the duck. Itai wants one or two more textured inputs.

| status | artifact | what to pull |
|---|---|---|
| LISTED | [Duck Figure Draft](https://claude.ai/code/artifact/653ae2b0-de4b-4ad2-9be6-892d2fc6a176) | the shipped Fig. 7, later draft |
| LISTED | [Duck Trimmed Figure](https://claude.ai/code/artifact/a367d7b4-d82b-4e18-8d45-88adffa394b7) | trimmed frame strip variant |
| LISTED | [Duck Camera Sheet](https://claude.ai/code/artifact/e8e3d4de-044d-4a60-9a3c-d9ddacb789ba) | camera contact sheet |
| LISTED | [Duck Figure Bench](https://claude.ai/code/artifact/7e7a4f55-6a61-465e-89de-d13cf8a07d7d) | bench of variants |

**Gap.** Same blocker as the gallery: a new textured mesh needs its own Kling video
and a fresh fit. No artifact for a second textured input.

## 5. MeshNCA ablation — four frames in, method run per frame

Itai's ask: give MeshNCA four input frames rather than one, run it per frame, and show
the four results on the right.

| status | artifact | what to pull |
|---|---|---|
| **HAVE** | [MeshNCA Conditioning Ablation](https://claude.ai/code/artifact/8f02c43d-e570-48a6-a9fd-db906078413c) | the 4x4 grid, the coverage measurement, and a drafted replacement for the MeshNCA paragraph |

**Done, 2026-08-31.** MeshNCA was refit from scratch on each of the four frames the
comparison figure prints (`spot_lava` 16 / 53 / 98 / 150), then rolled out 150 steps and
sampled at those same four instants. Jobs 2237644/45/46, 4000 epochs each on A40. The
frame-150 arm is not a new run: it is `results/ours_spot_lava`, the model the paper
already prints.

The result is sharper than the ask. Read **across** a row and nothing advances. Read
**down** the t=16 column and the output runs from bare grey to fully molten, so what the
render shows is the conditioning frame, not the timestep. Measured over all 150 frames,
the molten share of the silhouette moves **+0.52** for the reference video against
**+0.03 / +0.03 / +0.03 / -0.07** for the four rollouts. Span is the *wrong* statistic
here and should not be quoted: a stochastic automaton jitters, so the rollouts wander
0.04-0.19 without going anywhere. Net change, first third to last third, is the honest one.

Why it cannot do otherwise, from their code rather than from assertion: `train_image.py`
draws `step_n = randint(15, 25)` and applies the same appearance loss to whatever state
falls out, with a pool carrying states across epochs. Stationarity **is** the training
objective, and the config says the same in one line (`condition: null`, a single
`target_images_path`). That turns "it never observes the progression" from a concession
into a property of the method.

**The figure to drop in** is the draw.io page, not the PNG:

```
baselines4d/figs/dynamesh_supplementary_meshcna_diagram.drawio     <- use this
baselines4d/figs/mnca_conditioning_ablation.png                    <- same layout, preview only
```

Layout: reference-video film cell on the left, that model's four timesteps to the right, one
row per conditioning frame. No per-cell captions and no per-row labels; the film cell
carries the row identity, as in Fig. 8.

Built by `build_mnca_ablation_drawio.py` at the Comparison figure's own scale, with every
style constant copied from its `less_space` page rather than invented, so the two pages
match when printed together:

| | Comparison figure | this page |
|---|---|---|
| cell | 349--368 x 339--362 | 355 x 350 |
| labels | `fontSize=75; Times New Roman` | same |
| title | `fontSize=65; Comic Sans MS` | same |
| divider | `dashed=1; dashPattern=8 8; strokeWidth=7` | same |

Page is 1978 x 1798 against Fig. 8's 2860 x 1870: same scale, narrower because this figure
is 5 columns wide, not 8. Images embed at 1000 px into 355 px boxes (Fig. 8's own cells are
480 px in a 351 px box), so it prints sharper than the original.

Other paths: renders `baselines4d/outputs/meshnca_abl_f{016,053,098}` plus `meshnca_ours`
for f150; conditioning exemplars `MeshNCA/data/textures_ours/spot_lava_abl_f*.jpg`.

**Camera and shadow are Fig. 8's, recovered not chosen.** The cells are rendered at
**diagA (yaw 45, elev +25)**, an unseen view, which is what Fig. 8's caption claims: its own
MeshNCA cell matches our diagA render at silhouette IoU **0.982**, against 0.68 / 0.54 / 0.34
for the other three views. The ground shadow comes from guanc's
`geometry-def/dynamesh_render/add_house_shadow.py`, **imported** by the builder rather than
reimplemented, so squash 0.14 / shear 0.34 / blur 12 / opacity 0.22 cannot drift away from
the gallery and flicker figures.

**The geometry was checked, because the cells look odd.** At diagA the cow reads as two
overlapping bodies. That is the shape, not a broken mesh: MeshNCA's diagA silhouette matches
our own DynaMesh render of the same mesh at **IoU 0.990**, and Fig. 8's own DynaMesh cell
matches ours at 0.996. A trap while checking this: `view_spot_lava_27m_diagA/frames/*.png`
is a **two-panel** image (frozen | ours) with a 28 px label bar, so comparing it whole gives
a misleading IoU of 0.27; crop to the right panel first. One real artifact, and it is not
ours: a thin dark seam across the neck from MeshNCA's own renderer on the subdivided mesh.
It is present in the f150 arm, which *is* the model the paper already publishes, so it is
constant across all four rows and cannot bias the ablation.

**The four timesteps were recovered, not chosen.** They come from the shipped Comparison
draw.io (`less_space` tab): its filmstrip images match `frames_from_video` 16 / 53 / 98 /
150 by both lava-pattern IoU and pixel MAE, and those are columns 3, 8, 14, 21 of the
21-frame SV4D2 clock in `inputs/spot_lava/sv4d2/meta.json`. One trap for anyone redoing
this: the first filmstrip cell holds **two stacked images**, frame 1 underneath and hidden,
frame 16 visible on top. The conditioning crop was recovered the same way,
(336,312,624,600) -> 256px -> JPEG q95, which reproduces the shipped `spot_lava_f150.jpg`
at MAE 0.000, so the three new arms differ from the paper's arm in exactly one variable.

Not in the artifact, left for Itai's call: MeshNCA does not reproduce the crack *structure*
either. The exemplar's thin sharp veins come back as blobby patches, and the f150 arm holds
0.83 molten against the target's 0.69. That is a fidelity complaint, separate from the
temporal argument, and mixing the two would dilute the point.

---

# Tables

## 6. Components off/on ablation

Itai's bullet order is kept: (a) videos, (b) texels.

### 6a. Evaluated on videos, so the all-components row matches the paper

| status | artifact |
|---|---|
| **NONE** | no artifact exists |

The compute is done, the aggregation and the page are not. Per-object video-space
metrics exist for every arm: `out/FULLRATE_R19` 42, `out/FULLRATE_R37` 42,
`out/FULLRATE_W11` 42, `out/FULLRATE_R31` 24, `out/FULLRATE_R31M` 24. But every
`out/component_ladder*.json` on disk is dated Aug 24-29 and is the 24-object texel
one, so no video-space ladder was ever assembled.

The row Itai actually wants lands at **24.89 / 0.798** against Table 1's 24.90 / 0.798.

### 6b. Evaluated on texels

| status | artifact | what to pull |
|---|---|---|
| **HAVE** | [Supplementary Ablations](https://claude.ai/code/artifact/12e9de5c-604b-46b0-bd2d-81e44f2bd9fb) | Supp Table 1, the adapters-first ladder, in both its 24- and 42-object forms |

That page's **Supp Table 1** is the ladder for this bullet. It exists twice on the
page. The 42-object version is the one the paper prints as Table S2; the 24-object
version is the earlier run, kept for history. Do not mix the two object sets.

**Supp Table 1, "Adapters first, all five batches" — 42 objects. This is paper Table S2.**

| Configuration | CA | SA | Temp. | PSNR | SSIM | Flicker | Accel. | Drift | vs row above, flicker |
|---|---|---|---|---|---|---|---|---|---|
| Frozen TRELLIS.2 | x | x | x | 12.01 | 0.3263 | 0.01840 | 0.02740 | 0.3435 | — |
| + LoRA CA | ok | x | x | 23.61 | 0.6748 | 0.00725 | 0.01002 | 0.2092 | 42/42, p<1e-12 |
| + LoRA CA + LoRA SA | ok | ok | x | 24.80 | 0.7141 | 0.00707 | 0.00971 | 0.2046 | 27/42, p=0.088, n.s. |
| + LoRA CA + LoRA SA + Temporal | ok | ok | ok | 24.84 | 0.7149 | 0.00577 | 0.00593 | 0.2049 | 42/42, p<1e-12 |

**Supp Table 1, "Adapters first" — 24 objects, earlier run.**

| Configuration | CA | SA | Temp. | PSNR | SSIM | Flicker | Accel. | Drift | vs row above, flicker |
|---|---|---|---|---|---|---|---|---|---|
| Frozen TRELLIS.2 | x | x | x | 11.50 | 0.3363 | 0.02074 | 0.03102 | 0.3621 | — |
| + LoRA CA | ok | x | x | 23.47 | 0.7209 | 0.00854 | 0.01175 | 0.2288 | 24/24, p<1e-6 |
| + LoRA CA + LoRA SA | ok | ok | x | 24.76 | 0.7620 | 0.00844 | 0.01153 | 0.2247 | 15/24, p=0.31, n.s. |
| + LoRA CA + LoRA SA + Temporal | ok | ok | ok | 24.81 | 0.7630 | 0.00684 | 0.00701 | 0.2246 | 24/24, p<1e-6 |

All four rows of the 42-object table verified 42/42 on disk.

**Caption fix needed.** The page heads these columns "Temporal (texel-space)", but only
flicker, acceleration and drift are texel-space; PSNR and SSIM are image-space, taken
from each run's `final_eval.json` at the supervised view. As written a reviewer will
read "texel-space" as covering all five columns.

The page also carries a "Temporal first" ladder, the same four components in the
opposite order, useful for the attribution argument but not what this bullet asks for.

Older, 24-object-only version of the same ladder on its own page:
[Component Ablation](https://claude.ai/code/artifact/93443f39-82c7-4a34-a721-82cfc6722195).

## 7. Temporal attention ablation

Two learned variants against the parameter-free blend.

| status | artifact | what to pull |
|---|---|---|
| LISTED | [Learned vs Parameter-Free](https://claude.ai/code/artifact/d4857454-ff5a-4dea-a792-ef465eaffe3f) | rung 37 against parameter-free MCFM, newer of the two |
| LISTED | [Learned vs Parameter-Free, earlier](https://claude.ai/code/artifact/8d124889-8fcc-405e-ba52-44eea8e4597d) | earlier version, history |

**Which rung is which.** Read from `config.json`, not from run-directory names.

* *Temporal attention with LoRA* = **rung 37**. A second cross-attention branch
  initialised as a copy of the frozen TRELLIS cross-attention, LoRA-adapted, attending
  token j to token j across the window. Trained 42/42.
* *Spatio-temporal attention* = **rung 31**. One dual-branch registry pooling all
  W x N tokens into a single softmax over time and space, LoRA on the cross-attention
  weights. Trained 24/42.

**Open, and it decides the text.** Itai specifies 11 frames for the spatio-temporal
variant. Every temporal-attention run on disk is a 3-frame window. The 11 belongs to
MCFM's window, which is the parameter-free operator, not the learned attention, so the
text must not imply they were the same window.

**Symbols, per Itai's note.** Describe both variants with the paper's own symbols:
`z_t` conditioning tokens, `z̃_t` the blend (Eq. 1), `α_δ^i` the weights (Eq. 2),
`W' = W + ΔW` and `ΔW = (α/r)BA` (Eq. 3), `S` the structured latent, `D` the decoder
(Eq. 5), and the adapted attentions `CA` and `SA` (Eqs. S1, S2). For each variant state
which of those is duplicated, which is LoRA-adapted, and over how many frames the
attention runs.

## 8. LLM as a judge

| status | artifact |
|---|---|
| **HAVE** | [LLM Judge for Supplementary](https://claude.ai/code/artifact/f761de47-04b5-486e-b24b-7633aad7b060) |

**A first 42-object run was completed and is RETRACTED. Do not cite
`out/JUDGE/judge_table_42.tex` or `judge_table_pilot.tex`, and delete them before the
supplement is assembled.** Three faults, all in the questions rather than the judge:
effect fidelity was only ever asked at the training view, so the novel-view failure was
never tested; the geometry question asked whether the shape stayed *stable* rather than
whether it was *correct*, which a hallucinated-but-steady shape satisfies (L4GM runs
11-17% oversized at novel views and scored 4.65/5); and three of the four questions had
no reference at all. The tell is inside that table: the one question that had a
reference used 3.38 points of the 1-5 scale, the two without used 0.40 and 0.77.

**The protocol now.** Three questions, each with the reference its claim needs, every
one asked at all four cameras, three repeats, judge `gemini-3.6-flash`.

| | question | reference | scored as |
|---|---|---|---|
| Q1 | effect fidelity | the reference video | % of 5 yes/no questions passed |
| Q2 | temporal coherence | none needed | 1-5 |
| Q3 | geometry correctness | grey render of the input mesh, same camera | 1-5 |

Q1 follows Sining's protocol (LLM writes 5 yes/no questions, VLM answers them, score is
the pass rate) with **one deviation: the questions are generated from the reference video,
not the text prompt**, because 22 of the 42 objects have no Kling prompt on disk. It is
also the stronger reference, being what we trained against rather than what we asked
Kling for. The 5 slots are presence, appearance, spatial, temporal, restraint. Surface
adherence was dropped: the reference video itself scored 4.69 on it and frozen beat us.

**Report train, unseen-mean and the drop.** Absolute scores at the training view say
little, since every method was fit there. The drop is the finding.

**Pilot, 4 objects, 900 judgements. Not for the paper; the full 42 is not yet run.**

| method | Q1 train | Q1 unseen | Q2 train | Q2 unseen | Q3 train | Q3 unseen |
|---|---|---|---|---|---|---|
| Frozen TRELLIS.2 | 100.0 | 100.0 | 4.08 | 3.67 | 4.92 | 4.97 |
| MeshNCA | 56.7 | 50.6 | 2.25 | 2.17 | 5.00 | 4.94 |
| SV4D 2.0 | 100.0 | 91.1 | 4.58 | 4.39 | 5.00 | 4.36 |
| DreamGaussian4D | 83.3 | 50.0 | 1.83 | 2.97 | 5.00 | 4.22 |
| L4GM | 100.0 | 75.0 | 5.00 | 3.83 | 5.00 | 4.11 |
| **Ours** | **100.0** | **100.0** | **5.00** | **4.94** | 5.00 | **4.94** |
| *Reference video* | 100.0 | — | 4.92 | — | 5.00 | — |

**Two caveats that must reach the caption.** Frozen TRELLIS.2 also holds 100% on Q1 and
never drops, because it is handed the same mesh and is view-consistent by construction;
Q1 separates us from the video-based baselines, not from frozen, and Q2 is where frozen
loses. DreamGaussian4D's Q2 drop is negative, meaning it scores worse at its own training
view than at unseen ones, which needs checking on the full set before it is reported.

**LL3M as precedent, read from the PDF rather than assumed.** Its evaluation judge is
`gpt-4o` and it scores **code, not renders**: 34 bpy scripts across 17 objects with a
rubric prompt defining simple vs complex Blender operations. Gemini 2.0 Flash appears
only as the in-pipeline critic. Its Table 1 is a with-RAG vs without-RAG **ablation**,
not a baseline comparison, and there is no user study and no repeated trials. So LL3M
supplies the harness shape, a rubric prompt with definitions and examples, but no
protocol for scoring against baselines, and the 3-repeat averaging is ours.

**On disk.** `jobs/judge_vqa.py` (Q1, generate + answer), `jobs/judge_q23.py` (Q2/Q3),
`jobs/make_shape_refs.py` (the 168 grey references). Frozen questions in
`out/JUDGE/questions.json`, 210 of them, audited for shape leakage. Pilot records in
`out/JUDGE/pilot_vqa.jsonl` and `pilot_q23.jsonl`.

**Verification gates, all passed.** Reference video against its own Q1 questions: 100%.
Shape references against the pipeline's own render: mean IoU 0.9915, min 0.9604 over
168. Repeat std 0.199. Zero of 210 Q1 questions mention geometry. Q3's camera convention
was *solved* against the pipeline render, not assumed, reaching 0.9938 where every rival
convention scored below 0.70.

---

# 9. More info — things promised in the paper

Writing only, nothing blocking. The paper defers three things to the supplement that
must actually appear there:

* line 315, "the full formulas are in the supplementary material" — Eqs. S1/S2 cover this.
* line 381, "further training details are in the supplementary material".
* line 435, formal PSNR/SSIM definitions — Eqs. S3–S5 cover this.

**Inconsistency to resolve before anything else ships.** The main paper says an
eleven-frame window at line 372 and in the Fig. 4 caption, while Table S1 says
`[t-1, t, t+1]`, and Table S2's temporal row is numerically the W=3 run. The shipped
setting is W=3. Either correct the main text and Fig. 4, or re-run the headline at W=11.

---

# 10. Supplementary results page

Reference video, 3D result from the paper's angles, sections named after the figure
headlines.

| status | artifact | what to pull |
|---|---|---|
| **HAVE** | [DynaMesh Supplementary Results](https://claude.ai/code/artifact/ca94d1c8-10b8-4200-b4a2-4d3bd7cda44f) | the results page itself |
| LISTED | [Every Object, Every Method](https://claude.ai/code/artifact/1fbd4f9c-8cfe-440f-8aa4-d133a5fecec4) | the per-object pool it draws from |
| LISTED | [Review Response: Gallery Notes](https://claude.ai/code/artifact/f98158be-3245-4e34-aa39-d7f3338dbf10) | notes on what belongs in the gallery |

All three bullets are covered. **42 clips, seven sections, one per paper figure**, each
section titled with that figure's headline. Every grid carries the view across the top
and the method down the left.

| section | scenes | rows | clips |
|---|---|---|---|
| Fig. 1 · carries a video's effect onto a 3D object | aircraft | reference video, ours | 3 |
| Fig. 2 · gallery of results | hand, chair, pumpkin, airplane | reference video, ours | 12 |
| Fig. 5 · generalization | 6 unseen meshes | rorschach, lava | 12 |
| Fig. 6 · flicker | chair | frozen TRELLIS.2, ours | 4 |
| Fig. 7 · existing static texture made dynamic | duck | reference video, ours | 3 |
| Fig. 8 · qualitative comparison | spot | frozen TRELLIS.2, ours | 4 |
| Fig. 9 · failure case | vase | frozen TRELLIS.2, ours | 4 |

Cameras for the "angles presented in the paper": the supervised view is the camera the
reference video was generated from; the three unseen views are azimuth/elevation
(45, 25), (135, -20) and (225, 30).

**The page is visuals only, by instruction.** It must not become the supplement PDF in
HTML. No tables, no metrics, no method description. Formatting follows the plain
academic project-page convention (see `structurallydisentangled.github.io/page2.html`):
white ground, a horizontal rule between figures, bold figure headlines, scene name with
its prompt in italic underneath, and clips that autoplay and loop so the temporal effect
runs without being clicked.

**Removed on 2026-08-31, and must not come back.** Nine blocks that described the page
rather than the results: a "Scope" preamble, seven per-section intro lines ("one method
per clip", "Frozen TRELLIS.2 is not part of this figure", "a comparison figure"), and a
trailing "Reading the clips" note covering the left-half/right-half composite, the frame
counts, and how to scan rows against columns. The table of contents, the "Submission #92
/ Confidential review copy" banner and the seven paper figure captions were dropped in
the same pass, the captions because they duplicate the PDF.

> **Sharing caveat.** This artifact is shared by link and **viewers are pinned to an
> earlier version**, so anyone opening the existing link still sees the old text,
> including the "rows are what is shown" and "the submitted bundle carries all 65"
> lines. Re-share or unpin before circulating it.

Source clips are in `supp_bundle/videos`, 65 mp4s of which 42 are used; the page embeds
them as base64 so it is a single self-contained file.

---

# Summary of gaps

| item | blocker |
|---|---|
| Components off/on, video-space | data complete on disk, no artifact built |
| Gallery, car and robot | Kling videos plus a 30-epoch fit each |
| Existing textures, second example | same blocker as the gallery |
| Temporal attention, 11-frame question | Itai's call, see item 7 |
| W=3 vs W=11 in the main text | writing, but must be fixed |
| More implementation detail | writing only |

Everything else on Itai's list has an artifact behind it.
