# chair_moss — handoff

Everything Guan needs for the `chair_moss` object (furniture chair, moss effect),
3DV '27 / DynaMesh rung ladder. Written 2026-08-24.

---

## 1. The thing to look at

**Chair — Moss dailies:** https://claude.ai/code/artifact/96907e0e-5515-410b-bc67-2b9d490672b4

Index of all 15 objects: https://claude.ai/code/artifact/42db3ff0-0cc5-4ce7-92e1-0a03fec3b293

> **BEFORE SENDING:** both links currently serve a **pinned earlier version** to
> anyone but me. Move the share pin in the artifact UI first, or Guan sees stale
> tiles. Not a broken link — a silently old one, which is worse.

Page has four tabs — Training view, Diagonal A, Diagonal B, Diagonal C — each a
16-tile grid: ground truth, frozen TRELLIS.2, then all 14 rung arms.

---

## 2. The four videos behind that page

```
/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/
  chair_moss_train_grid.mp4    1.1M
  chair_moss_diagA_grid.mp4    1.2M
  chair_moss_diagB_grid.mp4    1009K
  chair_moss_diagC_grid.mp4    1.1M
```

Built 2026-08-23 05:22–05:30. These are self-contained — if Guan only wants
"the result", this is it.

Pull them down:

```bash
scp rajhansini@fe02.ai.cs.uchicago.edu:/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/GRID/chair_moss_*_grid.mp4 \
  ~/Downloads/ 2>&1 | tee ~/Downloads/scp_chair_moss.log
```

### Tile layout

16 tiles = ground truth + frozen TRELLIS.2 + 14 arms (rungs 27–34, each ± MCFM
temporal-only; **32 and 34 take no MCFM**, so 14 not 16).

Two things to say out loud when showing this:

- **The frozen tile is cropped, not re-rendered.** `render_arm.py` writes each
  frame as `frozen | adapted` side by side, so the grid takes the frozen tile from
  an arm's left half. Same frozen model, same camera, same frame indexing — it
  cannot drift out of step with the arms it's compared against.
- **Away from the training view there is no ground truth.** The Kling video is ONE
  camera. On the diagA/diagB/diagC tabs the GT tile shows the *training-view* frame
  and says so, dimmed, in its caption. It's there as a texture reference — what the
  effect should be doing at that instant — not as GT from that angle.

---

## 3. Supporting paths

| What | Path |
|---|---|
| GT Kling video, mesh, 150 extracted frames | `/net/projects/ranalab/rajhansini/TRELLIS.2/data/chair_moss/` |
| Per-arm renders — 154 dirs, 14 arms x 11 views | `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/view_chair_moss_<arm>_<view>/` |
| Grid builder | `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/build_grid_panel.py` |
| Per-arm renderer | `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/render_arm.py` |
| Texel-flicker metrics | `/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/texel_chair_moss_r29/`, `..._r30/` |

`data/chair_moss/` contains:
- `video/chair_moss.mp4` — the Kling source
- `frames_from_video/` — 150 frames
- `mesh/chair_moss_render_frame.obj`

Arm dir naming: `view_chair_moss_<rung>[m]_<view>`, where trailing `m` = +MCFM
and `<view>` is one of `train`, `diagA`, `diagB`, `diagC`, `u0`–`u6`.
Arms present: 27, 27m, 28, 28m, 29, 29m, 30, 30m, 31, 31m, 32, 33, 33m, 34.

---

## 4. Baseline: SV4D2, angle-matched

```
/net/projects/ranalab/rajhansini/baselines4d/outputs/sv4d2_ours/chair_moss/
```

5 view dirs, 105 PNGs, 21 frames at 576px. Job 2199402, COMPLETED
2026-08-22T18:24. Cameras were moved onto **our** rig so the comparison is
angle-matched: az `[0,45,135,180,225]`, elev `[0,25,-20,0,30]` →
`view_train`, `view_diagA`, `view_diagB`, `view_f180`, `view_diagC`.

**Caveat worth stating rather than hiding:** SV4D2's *trained* sampling pattern is
az `[0,60,120,180,240]` at elev 0. Our off-pattern azimuths and non-zero
elevations are in-API but out-of-distribution, so SV4D2 quality may drop. That is
a property of the baseline under angle-matching, not of our cameras. The
alternative was worse — on SV4D2's own grid only yaw 0 overlapped ours, and that
one is SV4D2's *input* view, so there was no fair novel-view cell at all.

`view_train` is the input view re-encoded, not a generated view — don't score it.

Same object on SV4D2's native grid, if Guan wants the in-distribution version:
`/net/projects/ranalab/rajhansini/baselines4d/outputs/sv4d2/chair_moss/`
(5 views, 105 PNGs, az 0/60/120/180/240 — job 2197753 shows FAILED in slurm, but
that was only the postprocess glob dying after sampling finished; frames were
recovered by `postprocess_sv4d2.sh` and are complete).
