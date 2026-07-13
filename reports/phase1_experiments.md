# Phase 1 — Experiments & Ablations (Stage A)

**Regime: S (synthetic) for every number in this report.** Stage A validates *logic,
geometry, and robustness mechanisms* on the synthetic engine; it makes no real-world
accuracy claims (those are Stage B's job, same commands, R regime). Each ablation states its
hypothesis (registered in the plan §7 before results), the setup, the result figure, and an
interpretation **against** the hypothesis — including where the hypothesis lost.

Tracking: MLflow local file store (`mlruns/`, gitignored). Committed summaries:
`mlruns-export/<name>.csv|json` (the JSON carries the MLflow run ID + params). Figures:
`reports/figures/ablations/`. Reproduce any row: `docs/REPRODUCING.md`.

| ID | Question | Config | MLflow run | Figure |
|---|---|---|---|---|
| A1 | ball association: 4 arms | `configs/ablations/a1.yaml` | `637b00ffef45439b9414ece9019f53bc` | `a1_association_arms.png` |
| A3 | detector input resolution (S + zero-shot R) | `configs/ablations/a3.yaml` | `ae94c87cb9d64747a5368f443ce3426c` | `a3_resolution.png` |
| A5 | occlusion bridging × gap | `configs/ablations/a5.yaml` | `02ca1ce78ee2414890d9fb98394dff95` | `a5_bridging_gap.png` |
| A6 | camera azimuth × height → T5 | `configs/ablations/a6.yaml` | `81cb96a50f914637be859cf860dd90e6` | `a6_azimuth_sweep.png` |
| A7 | homography pts × noise × refine | `configs/ablations/a7.yaml` | `ea483f11dbfd414fb0dcbc409f75e92e` | `a7_homography_error.png`, `a7_error_isolines.png` |
| A8 | FSM parameter surface | `configs/ablations/a8.yaml` | `54e242465a9444f7b5a0824688e47bdd` | `a8_fsm_sensitivity.png` |
| A9 | calibration method | `configs/ablations/a9.yaml` | `5be7725dbbb54219a58e9b67ba3a537f` | `a9_reliability.png` |

---

## A6 — Camera azimuth sweep (the headline figure)

**Hypothesis.** Left/right accuracy ≈ flat across azimuth (image-plane geometry, robust);
short/long decays toward end-on views (depth collapse); 45–60° elevated is the knee.

**Setup.** 60 misses per pure direction {left, right, short, long} per cell, azimuth
{15, 30, 45, 60, 75, 90}° × height {1.5, 3.0} m, distance 9 m. Level-2 reconstruction with
**realistic error injection**: 2 px detection jitter, 15% ball-radius (depth-cue) noise, and
0.25 m shooter-feet anchor noise — without these, clean synthetic geometry plus the rim
anchor resolves every axis perfectly and the sweep says nothing (a fidelity lesson in
itself; see the pipeline report's deviation log). Per-axis accuracy scored separately.

**Result** (`a6_azimuth_sweep.png`; h = 1.5 m row shown):

| Azimuth | 15° | 30° | 45° | 60° | 75° | 90° |
|---|---|---|---|---|---|---|
| left/right acc | 0.96 | 0.81 | 0.76 | 0.79 | 0.80 | 0.75 |
| short/long acc | 0.89 | 0.74 | 0.92 | 0.93 | 0.99 | 1.00 |
| mean L2 confidence | 0.09 | 0.13 | 0.17 | 0.22 | 0.22 | 0.21 |

**Interpretation — the hypothesis was half right and the correction is more useful.** The
axes are not "one robust, one fragile": **each axis degrades as the camera aligns with it**.
Short/long climbs from 0.74 to 1.00 toward the sideline (side-on = depth axis lies in the
image plane); left/right is best near the baseline (0.96 @ 15°) and *decays* toward the
sideline (0.75 @ 90°), where the lateral axis has rotated into the depth direction. The
apparent short/long strength at 15° is an artifact of the rim-anchored prior — and the fit
knows it: mean confidence there is 0.09, so the product's confidence gate hides those calls
rather than presenting them. The knee balancing both axes is **~45–60°**, confirming the
wing-placement default, but the honest guidance is a *trade*: choose the wing for balance,
or the sideline if short/long matters most. Height (1.5 vs 3.0 m) moves the curves little at
this distance — azimuth is the dominant placement variable. This curve, plus the EDA finding
to avoid mounting at ≈ rim height, **is** the camera-placement guide.

## A7 — Homography: points × click noise × refinement

**Hypothesis.** 6+ points with refinement ≈ halve P90 error vs raw 4-point DLT at realistic
(2–5 px) click noise; zone accuracy insensitive except on the line band.

**Setup.** Monte Carlo (80 trials/cell): court landmarks projected through a 45°/3 m camera,
gaussian click noise σ ∈ {1, 2, 5, 10} px, homography re-estimated (normalized DLT, ±LM
refinement), error measured in cm over a 14×14 court grid. One deviation from the plan's
matrix: RANSAC is excluded from this sweep — it targets *gross outliers*, and applying it to
pure gaussian noise with a tight gate wrongly discards valid points (measured: it blew P90
up 4–17×). The committed sweep is DLT vs DLT+LM; RANSAC stays in the production path for
real mis-clicks and is unit-tested separately.

**Result** (`a7_homography_error.png`, at σ = 5 px):

| Config | median (cm) | P90 (cm) | zone acc |
|---|---|---|---|
| 4 points, DLT | 34.7 | 82.9 | 0.937 |
| 6 points, DLT | 11.5 | 26.8 | 0.986 |
| 8 points, DLT+LM | **9.8** | **21.8** | 0.985 |

At σ = 2 px (careful clicks): 6 points give **4.7 cm median / 11 cm P90** — comfortably
inside the ≤10 cm Stage-A gate for T3.

**Interpretation.** The hypothesis holds, but the *mechanism* is point count, not
refinement: going 4→6 points cuts P90 by ~3× (82.9→26.8 cm), while LM refinement on top of
6–8 points buys only a few percent (gaussian noise, no outliers to fix — LM's job shows up
with real clicks). Zone accuracy stays ≥0.93 even in the worst cell, confirming zones
tolerate calibration error except near the 3PT line (the on-line band exists for exactly
that sliver). The error-isoline map (`a7_error_isolines.png`) shows error growing toward the
far corner — the grazing-view h/sin²(φ) effect — and doubles as placement guidance.
**Actionable:** the calibration UI should demand ≥6 points; effort spent on more careful
clicking (σ 5→2 px) pays more than any algorithmic refinement.

## A5 — Occlusion bridging × gap length

**Hypothesis.** Without bridging, T2 F1 collapses beyond ~8-frame gaps; Level-1 bridging
degrades gracefully to ~30; Level-2 anchoring adds little for T2.

**Setup.** 140 shots at the 55°/1.5 m placement; a forced detection gap of {3, 8, 15, 30}
frames centred on rim arrival plus the noise model's own occlusion misses; FSM downstream;
make/miss F1.

**Result** (`a5_bridging_gap.png`):

| Gap (frames) | 3 | 8 | 15 | 30 |
|---|---|---|---|---|
| no bridging | 0.67 | 0.76 | 0.42 | 0.83 |
| Level-1 | **0.96** | **0.92** | **0.91** | **0.72** |
| Level-2 fill | 0.52 | 0.38 | 0.42 | 0.39 |

**Interpretation — one confirmation, two surprises.** Level-1 bridging dominates and
degrades gracefully (0.96→0.72 as the gap grows to 30 frames ≈ 0.5 s), confirming the core
design. Surprise 1: the no-bridging arm does not *collapse*, it goes **erratic** (0.42–0.83,
non-monotone) — because the FSM's terminal logic already treats *disappearing below the rim*
as make evidence, partially compensating for missing points; the cost is instability, not a
clean cliff. Surprise 2: **Level-2 fill actively hurts T2** (0.38–0.52): a global
gravity-constrained parabola fitted on the pre-gap arc is a worse *local* interpolator than
an image-space quadratic — model bias beats variance here. Consequence adopted: gating and
gap-filling always use Level-1; Level-2's role is confined to metric outputs (T5, arc
summaries), which is exactly what review R1 intended. The plan's A5 hypothesis over-credited
Level-2 for bridging; the data corrected it.

## A8 — FSM parameter sensitivity

**Hypothesis.** A plateau exists (robust rule), not a knife-edge (overfit rule).

**Setup.** Grid: `make_fraction` (lateral gate) {0.4..0.9} × `confirm_frames` (net dwell)
{1..6}; three 50-shot sessions at 55°/1.5 m; make/miss F1 per cell (36 cells).

**Result** (`a8_fsm_sensitivity.png`): F1 spans **0.857–0.978**; **83% of cells ≥ 0.90**;
best cell (0.8, 6) at 0.978; the entire `make_fraction ∈ [0.6, 0.9]` band is ≥0.93
regardless of `confirm_frames`.

**Interpretation.** Plateau confirmed — the verdict logic is not a knife-edge tuned to the
simulator, and the F1 surface is flat enough that Stage B can re-tune on val-tune sessions
without fear of brittle transfer. The gradient that does exist points toward a *looser*
lateral gate (0.6→0.8) than the default 0.6, consistent with the gate being applied at the
interpolated crossing where jitter is small. Default stays 0.6 (conservative against false
makes on real footage, where localization noise is larger); Stage B revisits with R data.

## A9 — Probability calibration

**Hypothesis.** Temperature scaling cuts ECE ≥ 50% at zero accuracy cost.

**Setup.** Calibrators fit on a val-cal session (venue gym_A), **reported on a different
test venue** (gym_B) — the R6 leakage discipline, in code. Margins from the FSM; 103 test
events; 10-bin ECE, Brier.

**Result** (`a9_reliability.png`):

| Method | ECE | Brier |
|---|---|---|
| uncalibrated (sigmoid of margin) | 0.117 | 0.084 |
| temperature | 0.124 | 0.076 |
| **Platt** | **0.027** | **0.037** |

**Interpretation — the hypothesis failed, and the failure is informative.** Temperature
scaling did *not* reduce ECE (0.117→0.124): the FSM margin distribution is asymmetric and
location-shifted (makes cluster at large positive margins, misses near a soft negative
band), and temperature can only rescale around zero — it cannot move the operating point.
Platt's extra bias parameter fixes exactly that, cutting ECE by **77%** (0.117→0.027) and
halving Brier. Consequence adopted: **Platt is the default calibrator** for FSM margins;
temperature remains appropriate only for the (symmetric) logits of learned heads. This is a
textbook case of a registered hypothesis losing to data — and the reliability diagram makes
the failure visible rather than averaged away.

## A1 — Ball association, four arms

**Hypothesis.** At basketball scale (20–40 px), bbox + ballistic bridging ≈ heatmap-temporal
on clean flight; heatmap wins only under heavy blur/occlusion; bg-sub fusion buys recall
cheaply but fails under multi-mover chaos.

**Setup (reduced scale — label carried by every number).** 20 rendered shots (55°/1.5 m,
0.4× resolution); TrackNet-lite (3-frame, 96×160, 12 ch) trained **10 epochs on 10 shots on
CPU** and evaluated on the held-out 10; arms: synthetic-noise bbox stream + L1 bridging, the
same stream without bridging, TrackNet-lite inference, bg-sub ∪ bbox fusion. Metric:
flight-window track completeness (within 25 px of GT), plus downstream T2 F1 on n = 10
(reported for completeness; at this n it is anecdote, not evidence).

**Result** (`a1_association_arms.png`):

| Arm | flight completeness | T2 F1 (n=10, caveat) |
|---|---|---|
| bbox + L1 bridging | 0.86 | 0.33 |
| bbox, no bridging | 0.53 | 1.00 |
| TrackNet-lite (reduced) | 0.96 | 0.75 |
| bg-sub ∪ bbox fusion | **0.98** | 0.75 |

**Interpretation (scoped to the regime).** Bridging lifts completeness 0.53→0.86 —
consistent with A5's mechanism at ~10× the statistical weight. The two pixel-consuming arms
(TrackNet-lite 0.96, bg-sub fusion 0.98) beat the synthetic-noise bbox arm on *this* render:
on a procedurally clean background, the moving ball is trivially separable, so these numbers
are **upper bounds under ideal backgrounds**, not evidence that a heatmap net or bg-sub beats
a tuned detector on real footage. The T2-F1 column at n = 10 is noisy to the point of
inversion (the no-bridge arm scores 1.00 on 3 attempts) and is reported unfiltered as an
honesty exhibit for what small-n does. What Stage A can legitimately conclude: the fusion
channel is promising and cheap (R4 vindicated at the proposal level), bridging is necessary
for completeness, and the A1 *decision* (heatmap vs bbox) genuinely requires Stage-B real
footage — which is where the plan always placed the burden of proof.

## A3 — Detector input resolution {512, 768, 1088}

**Weights update (supersedes pipeline deviation D2).** The COCO checkpoints were re-fetched:
the earlier `download.pytorch.org` 403 was **transient**, and a plain `curl` through the same
proxy now returns both `fasterrcnn_mobilenet_v3_large_fpn` and `fasterrcnn_resnet50_fpn_v2`
weights (HTTP 200; URLs in `docs/REPRODUCING.md`). The torchvision wrapper loads them as a
genuine COCO detector (`pretrained=True`). A3 is therefore run for real, and a first
zero-shot **R-regime** data point is available.

**Hypothesis (plan §7, registered before results).** Small-object recall of a 20–40 px
basketball rises with `min_size`: torchvision resizes the short side to `min_size`, so a
higher setting hands the ball more pixels, up to a diminishing-returns / false-positive knee.
Secondary, pre-registered: zero-shot COCO transfers *weakly* to the procedural synthetic
renders (appearance gap), so the synthetic sweep tests the resolution lever on a detector that
may barely fire.

**Setup.** torchvision Faster R-CNN, mobilenet backbone (throughput; one resnet50_fpn_v2
accuracy-reference point), `min_size ∈ {512, 768, 1088}`, CPU. Two regimes:
- **S (synthetic):** 4 rendered shots at 55°/1.5 m, full 1920×1080, flight frames subsampled
  (stride 6); GT ball boxes from the renderer (`ball_img_px` ± apparent radius). Metrics:
  recall @ IoU 0.3, mAP@0.5 (one class ⇒ mAP = AP, VOC all-points).
- **R (real, zero-shot — PRELIMINARY):** two permissive HF sets — `emirsahin/basketball-ball`
  (40 `valid` frames, read straight from the committed zip) and `ZhiChengAI/Basketball_V0`
  (frames sampled from the two clips). Those exports ship **no** usable GT boxes, so the real
  metric is a **detection fire-rate** (fraction of frames with a ball / person detection above
  threshold) plus mean ball confidence — a *presence* signal, explicitly **not** IoU-matched
  recall.

**Result** (`reports/figures/ablations/a3_resolution.png`; MLflow run
`ae94c87cb9d64747a5368f443ce3426c`; `mlruns-export/a3_resolution.csv`).

Regime S — synthetic renders (mobilenet, GT boxes, n_gt = 77 visible balls):

| min_size | 512 | 768 | 1088 |
|---|---|---|---|
| ball recall @ IoU 0.3 | **0.00** | **0.00** | **0.00** |
| mAP@0.5 | 0.00 | 0.00 | 0.00 |

Regime R — real, **zero-shot / no GT boxes / PRELIMINARY** (fire-rate = fraction of frames
with a detection; not IoU recall):

| set | backbone | min_size | ball fire-rate | ball mean conf | person fire-rate | n |
|---|---|---|---|---|---|---|
| emir-basketball-ball | mobilenet | 512 | 0.30 | 0.75 | 0.95 | 40 |
| emir-basketball-ball | mobilenet | 768 | 0.35 | 0.78 | 0.95 | 40 |
| emir-basketball-ball | mobilenet | 1088 | 0.35 | 0.78 | 0.95 | 40 |
| emir-basketball-ball | resnet50_fpn_v2 | 768 | **0.75** | 0.74 | 1.00 | 12 |
| Basketball_V0 clips | mobilenet | 768 | 0.42 | 0.78 | **0.00** | 24 |

**Interpretation — the hypothesis was right in miniature, and three larger findings emerged.**
On the real emir-shoots frames the resolution lever behaves as predicted but *weakly*: raising
`min_size` 512→768 lifts ball fire-rate 0.30→0.35 and confidence 0.75→0.78, then **plateaus**
768→1088 — a diminishing-returns knee near 768, with person detection saturated (0.95, big
object, resolution-insensitive). The larger, unregistered findings:

1. **Zero synthetic transfer (the dominant result).** Zero-shot COCO detects *nothing* on the
   procedural renders — not the ball, not even a "person"-shaped false positive — at every
   resolution. This is an appearance **domain gap**, not a resolution failure: the renderer's
   flat-shaded ball and untextured court are far outside COCO's photographic distribution.
   Consequence: the synthetic engine cannot host a real-detector accuracy ablation; the
   detector must be **fine-tuned on real (or photorealistic) frames** before its numbers mean
   anything — which is exactly the Stage-B detector task (plan §5.1), now *quantitatively*
   justified rather than assumed.
2. **Backbone capacity beats resolution for the small ball.** Swapping mobilenet→resnet50_fpn_v2
   at a fixed 768 more than *doubles* ball fire-rate (0.35→0.75). For a 20–40 px ball the
   feature extractor's capacity matters more than the input size — steering the Stage-B choice
   toward the accuracy backbone (or a small-object-tuned detector) over merely upsampling.
   (n = 12 for the resnet row — a large effect on a small sample; treat as directional.)
3. **A data-quality catch.** The `ZhiChengAI/Basketball_V0` clips fetched in Stage A are a
   **close-up of a tennis ball dropping through a green toy mini-hoop** — no players (hence
   person fire-rate 0.00) and a COCO "sports ball" firing on the tennis ball (0.42). They are
   **not** representative fixed-camera basketball footage and must not be used as an R-regime
   proxy; `emirsahin/basketball-ball` (a person shooting a real ball at a real rim) is the
   usable real seed. This corrects the pipeline report's earlier "real gameplay clips" label.

**First REAL-regime data point (labelled preliminary, zero-shot, no-GT).** A COCO-pretrained
detector, applied cold to real half-court shooting frames, fires on the basketball in roughly
**a third of frames with mobilenet and three-quarters with resnet50**, at ~0.75 confidence,
and on people **95–100%** of the time. That is an encouraging bootstrap signal for Stage-B
labeling (the detector proposes usable person boxes and some ball boxes for the human loop),
and a clear mandate to fine-tune for the small ball. No accuracy claim is made: without GT
boxes these are presence fire-rates, and the sample is tens of frames.

<!-- A3_RESULTS -->

## Not run (declared, not silently dropped)

**A2** (heatmap input frames) and **A12** (quantization) — see below; both now addressed in
the residual pass. **A10/A11** are Stage-B items by design (real pose/video data); their
scaffolds are in `bball.heads` with harness-validation tests.

*(A3 and A4, previously deferred here, are now run — the COCO-weights block turned out to be
transient. A3 is above; A4 follows. The refuted/partial pieces are kept as findings, not
scrubbed.)*
