# Phase 1 Final Report — Basketball Shot Tracking from a Fixed Phone Camera

**Scope:** consolidated account of Phase 0 (research), planning + adversarial review, and
Phase 1 Stage A (pipeline + experiments in a CPU-only container). Stage B (real footage,
detector fine-tuning, cross-venue numbers) has a ready runbook and starts when data arrives.
**Regime discipline:** every number below is tagged **S** (synthetic) or **R** (real).
S validates *logic and geometry*; only R can validate perception. That separation is
enforced, not aspirational — see A3 below for the experiment that proves why it matters.

**Reading order for a reviewer:** `README.md` → this report → `docs/PROJECT_PLAN.md` (the
contract) → `docs/PLAN_REVIEW.md` (what adversarial review caught) → per-stage reports
(`phase0_research.md`, `phase1_eda.md`, `phase1_pipeline.md`, `phase1_experiments.md`) →
`notebooks/demo.ipynb` (`make demo`).

---

## 1. What was built and why it is shaped this way

A phone films a half court from a fixed wide angle; the system counts makes/misses and maps
every shot to a court location, with tiered stretch goals (shot type, miss direction, make
quality). The spine is **detect → track → lift → classify**, and two structural facts carry
the design: the fixed camera makes court registration a **one-time homography**, and
record-then-process makes compute a **budget, not a wall**.

**Phase 0 changed real decisions before any code existed** (`reports/phase0_research.md`):
the Ultralytics YOLO ecosystem was excluded outright (their AGPL-3.0 position extends to
self-trained weights), pushing the ship path to Apache-2.0 CNNs (RTMDet/YOLOX family) and
torchvision (BSD) for bootstrap; the famous "97% vs 53%" heatmap-vs-bbox ball-tracking gap
turned out to be shuttlecock-scale with an untuned baseline (97.51 vs 57.82 in the official
repo), demoting heatmap-temporal from assumption to hypothesis (ablation A1); Apple's
`VNDetectTrajectoriesRequest` + native pose emerged as a zero-third-party-weights MVP path
for Phase 2; NEX Team patents (US11594029, US11380100) were flagged, and the make/miss rule
was deliberately designed to be mechanically distinct from net-motion-based detection.

**Adversarial review caught two real correctness errors before implementation**
(`docs/PLAN_REVIEW.md`): the v1.0 plan quietly assumed a single camera can observe an
airborne ball's 3D position (it cannot — a homography lifts ground-plane points only), and
its shot-attempt definition excluded airballs. The fixes shaped the architecture: near-rim
event logic is **rim-normalized** (predicates relative to the projective image of the rim
circle), trajectory fitting is split into an image-space level for association/bridging and
a confidence-gated 3D level for metric outputs only, and the review also added the classical
baseline a fixed camera demands (background subtraction).

## 2. Stage-A results (all S unless marked)

| # | Question | Result | Verdict vs. registered hypothesis |
|---|---|---|---|
| A7 | How accurate is manual homography? | **4.7 cm median / 11 cm P90** at 2 px clicks with ≥6 points; 34.7 → 9.8 cm as 4 → 8 points at sloppy 5 px clicks; zone accuracy ≥ 0.99 | Confirmed — with a twist: **point count, not LM refinement, is the lever** |
| A8 | Is the make/miss FSM a knife-edge? | F1 **0.86–0.98 across the whole 36-cell parameter grid**; 89–96% batch accuracy over 4 camera placements | Confirmed — a plateau, i.e., robustness, not overfit tuning |
| A5 | Does ballistic bridging survive rim occlusion? | Level-1 (image-space) F1 **0.96 → 0.72** over 3 → 30-frame gaps; no-bridging erratic; **Level-2 (3D-informed) fill made T2 *worse*** | Half-refuted — L2's demotion to metric-outputs-only, made empirical |
| A6 | Where should the camera go? | **The axes trade**: left/right accuracy 0.96 @ 15° → 0.75 @ 90°; short/long 0.74 @ 30° → 1.00 @ 90°; **45–60° elevated balances both** | **Refuted** (plan expected L/R ≈ flat) — the refutation *is* the placement guide |
| A9 | Are FSM margins calibratable? | **Temperature scaling fails** (ECE 0.117 → 0.124); **Platt cuts ECE 77%** to 0.027 on a held-out venue | Refuted for temperature, confirmed for Platt — negative result kept |
| A1 | Heatmap vs bbox vs bg-sub for the ball? | Four arms compared on synthetic; TrackNet-lite arm trained at reduced CPU scale (explicit small-n caveat in `phase1_experiments.md`) | Inconclusive by design at this scale — **Stage B decides on real footage** |
| A4 | How much player-tracker sophistication? | Greedy-IoU **0.715 HOTA** beat Kalman+IoU (0.578); ByteTrack-style recovery raised ID switches (36 vs 10) in the waypoint-walker sim | Surprise — hard direction changes break constant-velocity prediction; simplified single-alpha HOTA; real footage arbitrates |
| A3 | Does detector resolution buy small-ball recall? | COCO-pretrained detector: **0.0 ball recall on synthetic renders at every resolution**; on real images (R-zeroshot, no GT boxes): ball fire-rate **0.30–0.35**, person **0.95** | The synthetic half is the project's honesty argument made empirical: stylized renders cannot evaluate a real detector — regime labels exist for exactly this reason. The real half confirms fine-tuning is Stage B's first job |
| EDA | Anything the plan missed? | A camera at ≈ rim height (2.7–3.3 m) images the rim edge-on (ellipse ratio → 0) | New constraint, now in the protocol |
| Demo | Does it run end-to-end? | `make demo`: 6 synthetic shots through the full pipeline, **5/6 correct** with the miss narrated (point-blank visibility case) | G1 pass, zero downloaded weights |

Stage-A gates **G1–G5 all pass** (`reports/phase1_summary.md`): one-command demo, 120 tests
(geometry round-trips, nine scripted FSM scenarios including rattle-in, shooter's roll,
put-backs, lob-pass negatives, multi-ball distractors, anti-leakage guards), tracked
ablations with committed exports (`mlruns-export/`), cited physical constants, and a
figure → command map (`docs/REPRODUCING.md`).

## 3. Product-facing geometry: configurable zones

Because shot location is stored as a continuous court position, zone taxonomies are a
**view, not a measurement** (`docs/ZONES.md`, `src/bball/lift/zones.py`): presets
(basic 3-zone, extended with short/long-mid and deep-three, the classic spot chart),
parametric bounds in feet/metres, and freeform screen-drawn regions lifted through the
calibration homography into court space — camera-independent, retroactively re-bucketable.
Two details matter: **deep-three is an offset of the true 3PT shape** (arc + straight corner
segments — a radial threshold misclassifies the corner, and the tests pin the exact points
of disagreement), and every boundary composes with the A7 error field to score
**per-boundary reliability** — the app can warn that a specific drawn boundary is not
trustworthy from the current camera placement before the user trusts its stats.

## 4. Data protocol, simplified for a solo collector

`docs/DATA_PROTOCOL.md` now leads with a tiered minimum: **Tier 1** — one court, ~5 filmed
workouts (~3–4 h): vary tripod azimuth and ball (the free axes), call deliberate misses
aloud (free miss-direction labels), include shot-free negative blocks (honest false-positive
rates); **Tier 2** — 2–3 sessions at any second court, never trained on (the transfer
number). Ground truth needs almost no taping: painted landmarks are cm-accurate spots once a
single 2-minute measurement verifies the court's paint standard (the check that keeps
landmark ground truth from being circular), with the under-rim origin found by a string
plumb + 23 cm offset. Labeling is semi-automatic: the pipeline proposes, and
`scripts/review_events.py` steps a human through confirm/correct into the labels CSV.

## 5. Honest limitations, and what Stage B decides

No real-world accuracy claim exists yet — Roboflow/Drive were proxy-blocked, so R-regime
data is limited to zero-shot fire rates on fetched permissive image sets. Stage B, with the
runbook already committed: fine-tune the ship-path detector on protocol footage (+ the
license-gated downloader scripts), re-run every ablation's R regime with the same commands,
settle A1/A3/A4 on real data, fill the T4–T6 heads, implement the calibration drift monitor,
and produce the cross-venue headline table. Structural limits stay stated: short/long miss
direction is azimuth-dependent (A6 quantifies it; the app confidence-gates it), swish-vs-
rattle remains experimental with the audio channel as the designed upside, and the patent
posture (design-arounds documented, FTO before commercialization) is a maintained artifact,
not a footnote.
