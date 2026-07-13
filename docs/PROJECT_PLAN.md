# Project Plan — Basketball Shot Tracking from a Fixed Phone Camera

**Version:** 1.1 (post-review; see `docs/PLAN_REVIEW.md` for the adversarial review that produced it)
**Date:** 2026-07-13
**Companion documents:** `reports/phase0_research.md` (literature & SOTA survey), `docs/PLAN_REVIEW.md` (critique log)

---

## Summary

A phone films a half basketball court from a fixed wide angle; the system counts and tracks makes vs. misses, locates each shot on the court, and — in increasingly ambitious tiers — classifies shot type, miss direction, and make quality. The organizing spine is:

**DETECT** (ball, rim, player, court reference points) → **TRACK** (associate detections over time; bridge occlusions with ballistics) → **LIFT** (map image coordinates to real-world court coordinates via a one-time homography) → **CLASSIFY** (finite-state event logic and small learned heads on top of geometry).

Two structural facts do most of the work, and the whole design leans on them:

1. **The camera is fixed.** Court registration collapses from a per-frame learned-calibration problem (what KaliCalib/TVCalib/PnLCalib solve for moving broadcast cameras) to a one-time 4–8 point homography. Shot location becomes one of the *easiest* advanced features rather than a hard one.
2. **Recording and processing are decoupled.** iPhone slow-mo captures 1080p at up to 240 fps, but nothing forces inference at 240 fps. Record-then-process removes the real-time constraint from Phase 1 entirely and converts it into a Phase-2 latency *budget* (process a 30 s clip in ≤ clip duration).

Phase 1 is a research pipeline with experiments, tuning, and ablations, executed in two explicitly separated stages (§11): **Stage A** (this repository, CPU container: full pipeline, synthetic-data experiment engine, geometry/event ablations, small-scale training, evaluation harness) and **Stage B** (user GPU + self-collected footage: detector fine-tuning, real-footage benchmarks, cross-venue generalization). Every result is labeled with the stage and data regime that produced it; synthetic results validate *logic and geometry*, never perception accuracy claims.

---

## 0. Literature & Method Survey — Synthesis

Full survey with evidence tags and references: `reports/phase0_research.md`. The decisions it forces:

| Decision | Choice | Why (condensed) |
|---|---|---|
| Detector family | Apache-2.0 CNNs (RTMDet-tiny / YOLOX-S) for ship path; torchvision COCO-pretrained detectors for Stage-A bootstrap; D-FINE/RF-DETR as accuracy upside pending a Core ML path | Ultralytics AGPL-3.0 extends to self-trained weights and even internal R&D per their own licensing docs — the whole YOLO ecosystem, including its one-line Core ML export, is excluded. DETR-family Apache models lead on accuracy but have **no working public Core ML conversion** (open coremltools `grid_sample` bugs; RF-DETR issue #318). |
| Ball association | Per-frame small-object detection + explicit ballistic (parabola) occlusion bridging as primary; heatmap-temporal (WASB/TrackNet-style, MIT) as the ablation comparator and fallback | The famous "97% vs 53%" TrackNet-vs-YOLO gap is real but (a) it's 97.51 vs 57.82 per the official repo, (b) it's measured on a few-pixel shuttlecock, and (c) the YOLO baseline was untuned. A basketball at 20–40 px is a different regime; no published evidence says heatmap-temporal is *necessary* there. The genuinely temporal problem — rim/net occlusion — is attacked directly with a physics prior. This is now an empirical question our ablation answers, not an assumption. |
| Player tracking | ByteTrack-style Kalman + IoU with low-score recovery; no ReID | 1–3 players, fixed camera. TrackID3x3's BoT-SORT-ReID baseline only reaches 63–66 HOTA on fixed-camera 3×3 — sophistication does not buy robustness here; low-score box recovery (player half-hidden behind rim/another player) is the one add-on that pays. **Never point a SORT-family tracker at the ball** — track-kill-on-miss is exactly wrong under rim occlusion. |
| Court registration | One-time manual 4–8 point homography, RANSAC-refined; error budget derived analytically + Monte Carlo | Learned calibrators are wrong-sport, copyleft (KaliCalib CeCILL, PnLCalib GPL-2.0), or trained on non-commercial data. Click error of 2–5 px maps to ~5–20 cm on court at sane camera geometry — ample for zone classification (bins ≥ 1 m). |
| Make/miss | Geometric rim-local finite-state machine; learned event spotting (E2E-Spot lineage, BSD) only after ~1–2 k labeled shots accumulate | Geometry needs zero event labels, is auditable, and every failure is debuggable. Learned spotting at our data scale is not competitive (2025 SoccerNet BAS winner pretrained on 500+ matches for ~0.6 mAP@1). |
| Pose | Apple Vision native (deploy); RTMPose (Apache) for offline research if 19 keypoints limit features | Zero license/conversion risk on-device; RTMPose-m is 75.8 COCO AP at 90+ fps on CPU for the research side. |
| Shot type / few-shot | Dribble detection = signal processing on ball height + possession (not a learned model); custom types = prototypical head on frozen embeddings (pose-sequence primary; X-CLIP MIT as video-embedding alternative) | Pull-up vs catch-and-shoot reduces to "was there ball oscillation in the pre-release window" — a feature, not a network. VideoMAE weights are CC-BY-NC (trap: code MIT, weights NC); X-CLIP is MIT. Realistic few-shot band from adjacent benchmarks: ~80–92% for kinematically distinct classes at 10–40 clips/class. |
| Data engine | Grounding DINO/OWLv2 zero-shot seed → SAM 2.1 propagation (CVAT) → human verify → small detector → active learning | All Apache-2.0. 5–10 k labeled frames in days of part-time work; rim boxes are near-free (static per session). |
| Audio (make quality) | Log-mel window at vision-determined rim-arrival → frozen audio embedding (YAMNet/BEATs) → small head | **No academic prior work on basketball shot-outcome audio — a genuine gap.** Adjacent sports-acoustics results (table-tennis spin from bounce sound) support the mechanism. Cheap experiment, high novelty value; gated by vision timing, so it composes with the spine instead of replacing it. |
| Patent posture | Design-around notes from day one; FTO review before any commercialization | NEX Team US11594029 reads at abstract level on feet-at-release→homography location; US11380100 keys on net movement for result detection (our static rim-plane rule is mechanically different — document this). |

---

## 1. Problem Framing & Success Metrics

### 1.1 Task definitions (precise)

Let a **shot attempt** be an event where the ball leaves a player's hands with upward velocity on a flight whose apex is above rim height (3.048 m) and whose horizontal terminal direction is **toward the hoop** — ground-projected distance to the hoop decreasing over the descent, with the flight terminating within a generous hoop-centered radius or in backboard contact. Airballs therefore qualify (they are attempts, and usually "short"/"long" for T5); passes and dribbles do not. The known hard negative is the lob pass toward the rim region — measured explicitly on negative-block footage (§2.1). Put-backs and tip-ins count as *new* attempts: the event guard prevents double-counting a single flight, not genuinely new ones. The definition is operational — it is exactly what the trajectory layer tests. (Rewritten per review R3: the v1.0 definition required entering the rim neighborhood, which silently excluded airballs.)

| ID | Task | Input | Output | Tier |
|---|---|---|---|---|
| T1 | Shot detection | ball track + rim geometry | set of shot attempts with timestamps (release, rim-arrival) | MVP |
| T2 | Make/miss | shot attempt + ball track near rim | {make, miss} + calibrated probability | MVP |
| T3 | Shot location | shooter track + release timestamp + homography | court (x, y) in meters + zone ∈ {short-range, midrange, 3PT} (+ "on the line" band) | MVP |
| T4 | Shot type | pose sequence + ball track, 2–4 s pre-release | {pull-up, catch-and-shoot} (built-in); user-defined classes (few-shot) | Advanced |
| T5 | Miss direction | ball track near rim + rim geometry | {left, right, short, long} (multi-label allowed: e.g., short-left) + per-axis confidence | Advanced |
| T6 | Make quality | rim-ROI track + audio at rim-arrival | {clean swish, rim-in} + confidence | Experimental |

The **release moment** is defined as the first frame where ball-center-to-wrist distance exceeds 1.5× ball radius with positive vertical ball velocity, falling back to ball-center-above-head when pose is unavailable. All downstream tasks (T3's feet position, T4's window) key off this timestamp, so its tolerance is measured explicitly (±2 frames at 60 fps target).

### 1.2 Metrics — real-world units, honest denominators

- **T1:** event-level precision/recall/F1 with ±0.25 s temporal tolerance; per-session counts (the user-facing quantity: "you took 87 shots"); **false attempts per hour of shot-free footage** (negative blocks, §2.1) — precision measured only on shot-dense footage is an easy grader.
- **T2:** precision/recall/F1 per class; **probability calibration** — reliability diagram, Brier score, ECE (15 bins) before/after temperature scaling. The FSM emits a margin-based score (§5.4), not just a label; a make/miss counter that says "90% sure" must be right 90% of the time, and calibration is what makes downstream aggregates ("you shot 41% from midrange") trustworthy.
- **T3:** court-position error in **cm** (median, P90) against measured ground truth (tape-measured shot spots in collection protocol); zone accuracy + confusion matrix; % of shots flagged "on the line".
- **T4:** accuracy + macro-F1 (classes will be imbalanced); few-shot: accuracy vs. clips-per-class curve (5/10/20/40).
- **T5:** per-axis accuracy (left/right axis and short/long axis **reported separately** — a single camera collapses one depth axis, so aggregate accuracy would hide the structural weakness; §7 A6 quantifies it).
- **T6:** F1 + confusion matrix; audio-only vs. vision-only vs. fused.
- **Component metrics:** detector mAP@0.5 and small-object recall at IoU 0.3 in the rim-approach window (the ball is small; a loose box that maintains track continuity is worth more than a tight box that flickers — IoU 0.3 recall is the metric that predicts event-logic success); ball track completeness (% of flight frames with a position estimate, split by occlusion state); player tracking HOTA/IDF1 (when multi-player data exists).
- **Phase 2 gate metrics (deferred):** on-device latency per stage, energy per session, model size.

### 1.3 Targets

| Task | Stage-A gate (synthetic + any real clips) | Stage-B target (user footage, cross-venue held-out) | Basis |
|---|---|---|---|
| T1 | ≥ 99% event F1 on synthetic; zero double-counts under rebound stress tests | ≥ 97% | Synthetic is by-construction; real target from open-source repos' self-reports (88–97%) discounted for their unaudited methodology |
| T2 | ≥ 98% F1 on synthetic occlusion sweep | ≥ 92% F1 first pass, ≥ 95% after tuning; ECE < 0.05 | DeepDarts' honest cross-angle number (84%) and field reports (~90%+) bracket expectations; fixed camera removes DeepDarts' main degradation source |
| T3 | ≤ 10 cm median on synthetic; zone ≥ 99% | ≤ 30 cm median, zone ≥ 95% off the line band | Homography error model (§5.3): 2–5 px click error → 5–20 cm at sane geometry |
| T4 | pipeline + features validated on public action clips | ≥ 85% pull-up vs catch-and-shoot; few-shot per §0 band | X-CLIP few-shot benchmarks; dribble detection is near-deterministic when ball track is good |
| T5 | azimuth-accuracy curve produced (the deliverable IS the curve) | left/right ≥ 85%; short/long reported with confidence gating, no blanket target | Depth collapse is structural; we quantify instead of promising |
| T6 | harness + audio scaffold; no accuracy claim | exploratory report | No prior art exists; this tier is a research contribution, not a commitment |

Go/no-go: each tier proceeds only if the tier below hit its Stage-B target on the cross-venue set. A tier that misses ships as "experimental, off by default" or is cut — the report documents the failure honestly either way.

---

## 2. Data Strategy

### 2.1 The asset: controlled self-collection

The user controls the venue, camera, and session plan — the exact luxury commercial systems bought with 100k+ hand-shot training shots. The **collection protocol** (committed as `docs/DATA_PROTOCOL.md`, with a printable checklist) is designed to force generalization along the axes that will actually vary at deployment:

- **Sessions:** 20–40 min each, one camera setup per session (setup change ⇒ new session ID). Metadata sheet per session: venue, hoop type, ball, lighting, camera height/azimuth/distance (paced or tape-measured), fps, resolution.
- **Variation grid (minimum viable):** ≥ 4 venues (indoor gym ×2 with different floors, outdoor ×2 with different backboards) × ≥ 3 balls (leather indoor, rubber outdoor, worn/discolored) × lighting {daylight, dusk, indoor artificial} × **5 camera placements**: azimuth ≈ {0° (baseline under/behind hoop is excluded — occlusion pathology — so 15°), 30°, 45°, 60°, 90° (sideline)} at heights {1.5 m tripod, 2.5–3.5 m elevated}. The 45° wing placement is the recommended default (Phase-0 §10: both miss axes observable); the sweep exists to *measure* the recommendation, not assume it.
- **Per-session shot script:** ≥ 60 shots covering zones (paint / midrange / 3PT × left / center / right), deliberate makes AND misses (aim off on purpose: short, long, left, right — this yields miss-direction ground truth cheaply because the shooter *knows* the intended error), swish-hunting blocks and bank/rattle blocks for T6, pull-up and catch-and-shoot blocks for T4 (a rebounder/passer for C&S), 5 min of free play including rebounds and multiple-ball chaos, **≥ 5 min of explicit negative blocks** (dribbling/passing drills with zero shot attempts) plus a lob-pass block — the adversarial near-positives that make T1 precision numbers honest. For audio-critical (T6) sessions: prefer 60 fps normal-speed capture (slow-mo retimes video but records audio at normal rate — alignment risk), and start each session by bouncing the ball in frame once — a natural clapperboard for audio/video alignment verification.
- **Ground-truth spots for T3:** 9–12 floor positions marked with tape, positions measured from court landmarks with a tape measure (cm-level). Shots from marked spots give T3 its cm-error denominator.
- **Labels:** per shot — outcome, zone + marked-spot ID, shot type, miss direction (shooter-called, verified on video), make quality (swish/rattle, audio-verifiable in review), release frame (coarse). Labeling is **semi-automatic from day one**: the pipeline proposes events + timestamps; the human verifies/corrects in a purpose-built review UI (CVAT for boxes; a lightweight per-shot CSV editor for events). Every correction is a training example — active learning is the loop, not a stage.
- **Frame-level labels** (for detector fine-tuning): via the data engine (§0: zero-shot seed → SAM 2.1 propagate → verify), targeting 5–10 k frames with ball/rim/player boxes across the variation grid. Rim boxes: labeled once per session, propagated (static camera).

### 2.2 Public data — seeding, with license discipline

| Dataset | Use here | License note |
|---|---|---|
| Roboflow ball/rim sets (Basketball Video Analysis 6,076; Basketball and Rim 6,270; player-detection-3 654) | Detector seeding/fine-tuning; Stage-A real-image sanity checks | MIT / CC BY 4.0 — ship-safe; per-project badges re-verified at integration time |
| TrackID3x3 | Player det/track transfer + HOTA benchmark (closest camera analog: fixed-cam 3×3) | Data CC BY 4.0 |
| SpaceJam (~32.5 k action clips + joints) | Pose-action head pretraining (T4) | Repo MIT; clip provenance unstated — training use documented, no redistribution |
| Basketball-51 | Label-taxonomy reference; NOT trained on | Unofficial NBA-footage mirror — legally murky; read-only |
| DeepSportRadar | Methodology reference (ball-3D-from-diameter task) | Images CC BY-NC — excluded from anything that ships |
| FSD50K / AudioSet-pretrained YAMNet | T6 audio embedding | CC / Apache respectively |

Anything CC-BY-NC or unlicensed influences **method design only** — no weights trained on it ship. A `LICENSES.md` ledger records every dataset/model dependency and its role.

### 2.3 Splits — leakage discipline

- **Unit of splitting = session** (camera setup × venue × day). Never frames, never shots: adjacent frames are near-duplicates, and shots within a session share lighting/background/ball — random-frame splits would inflate every number.
- **Cross-venue test set:** ≥ 1 entire venue (all its sessions) held out from all training and tuning, touched only at evaluation milestones. This is the number the README reports.
- **Tuning split:** train/val split across the remaining sessions, stratified by venue; hyperparameter selection on val only.
- **Synthetic data:** split by generated *scene config* (court texture + camera pose + ball appearance bundle), mirroring the session discipline, so "memorize the background" is impossible in synthetic experiments too.
- **Few-shot (T4 custom):** episodic evaluation — support/query sets drawn from *different sessions*, so a prototype never matches on venue features.

### 2.4 Versioning

Raw video is immutable under `data/raw/<session_id>/` with a checked-in `manifest.yaml` (path, SHA-256, session metadata) — the data itself is gitignored; the manifest is the version. Derived artifacts (frames, tracks, features) are reproducible functions of (raw, config, code SHA) and are never hand-edited. Labels live in git as CSV/JSON (small, diffable, reviewable).

---

## 3. EDA — What to Inspect and Why

EDA here de-risks specific architecture decisions; each analysis has a consumer:

| Analysis | Feeds | Question it answers |
|---|---|---|
| Ball apparent size (px) distribution vs. court position, per camera placement | detector input resolution (A3); heatmap-vs-bbox framing | Are we in the 20–40 px regime the Phase-0 correction assumed, or smaller at 3PT-from-far-corner? |
| Motion blur: ball streak length per frame at 30/60/240 fps (line-fit on thresholded ball region) | fps recommendation; blur augmentation realism | Does 240 fps really deliver near-blur-free balls (Phase-0 claim), and what blur must augmentation cover for 30/60 fps fallbacks? |
| Occlusion timeline: fraction of flight frames with ball behind rim/net/backboard per camera azimuth | bridging design (A5); collection guidance | How long are the gaps the parabola must bridge — 3 frames or 30? |
| Rim pixel size + position stability across a session | rim auto-detection vs once-per-session labeling | Is per-session manual rim labeling (10 s of work) simply correct? |
| Class balance: makes/misses, zones, types per session | split stratification; loss weighting | Self-collected data will be maker-biased (people practice what they make) — quantify and correct via the shot script |
| Brightness/contrast/white-balance drift across sessions; court-line contrast | augmentation ranges; homography click reliability | Which lighting axes actually vary — augment those, not everything |
| Audio: waveform/spectrogram gallery around rim-arrival for swish/rattle/clank/backboard | T6 go/no-go | Are the classes separable by eye/ear at half-court recording distance? (If not visibly separable in spectrograms, T6 vision-only.) |
| Trajectory stats: release angle/speed distributions, apex heights, flight times | synthetic engine parameter ranges (fidelity); shot-attempt definition thresholds | Ground the simulator in measured reality, not textbook values |

Stage A runs every analysis on synthetic + whatever real clips exist, shipping the *tooling* (one command per analysis, plots to `reports/figures/`); Stage B re-runs on real footage — same commands, comparable plots, and the sim-vs-real distribution comparison doubles as the fidelity audit of the synthetic engine (§11 gate G4).

---

## 4. Baselines — the Simplest Thing That Could Work

Gains must be attributable; each pipeline stage gets a floor:

| Task | Baseline | Rationale |
|---|---|---|
| Ball detection (neural floor) | COCO-pretrained torchvision detector, class `sports ball`, zero fine-tuning | The "do nothing" floor; quantifies what fine-tuning buys |
| Ball detection (classical) | **Background subtraction (MOG2) + size/shape/temporal filtering** — the fixed camera is the strongest classical prior available, and the moving ball is the dominant small mover at wide framing | Without this row, "deep detector wins" has a hole where the obvious classical method should be; also a candidate-proposal channel fused with the detector (A1 third arm) |
| Ball association | Greedy nearest-neighbor linking, no motion model | Quantifies what the Kalman + ballistic layer buys |
| Trajectory | Apple `VNDetectTrajectoriesRequest` (documented as Phase-2 comparator; not runnable in a Linux container — Stage-B/Phase-2 item on-device) | The zero-model industry floor; free prior we may keep permanently |
| Make/miss | Naive rule from open-source repos: ball bbox center enters rim bbox while moving down ⇒ make | The pattern behind the "95–97%" self-reports; our FSM must beat it under occlusion/rattle stress, measurably |
| Shot location | Midpoint-of-player-bbox-bottom at release + raw homography, no refinement | Floor for A7 (refinement ablation) |
| Shot type | Logistic regression on two hand-crafted features: pre-release ball-height variance + possession duration | If a 2-feature model hits 80%, the temporal net must justify its complexity |
| Miss direction | Sign of (ball_x − rim_x) at rim-plane crossing in image space, no 3D reasoning | Exposes exactly when naive image-space logic fails (off-azimuth cameras) — sets up A6 |
| Few-shot | Nearest-class-mean on raw pose-sequence DTW distance | Floor under the prototypical head |
| Calibration | Uncalibrated FSM margins | Floor for temperature scaling |

---

## 5. Model & System Architecture

### 5.0 Pipeline (record-then-process)

```
clip.mp4 (+ audio)
  ├─ decode → frame stream (decimated 60 fps equivalent; full 240 fps only in rim-approach windows)
  ├─ [DETECT] ball / player / rim candidates per frame
  ├─ [TRACK]  players: Kalman+IoU w/ low-score recovery ──► shooter track
  │           ball: gated association → trajectory segments → ballistic bridging
  ├─ [LIFT]   one-time homography (session calibration) ──► court coords; rim-local 3D frame
  └─ [CLASSIFY] shot FSM ──► attempts, make/miss + margin score
                ├─ T3 feet-at-release → zone
                ├─ T4 pre-release window → type head
                ├─ T5 rim-local trajectory → miss direction
                └─ T6 rim-ROI + audio window → make quality
```

Design invariant: **no raw-pixel thresholds — all rules in rim-normalized or court-metric units, whichever is actually observable.** Pixel-space rules silently encode one camera placement. But a single camera does *not* observe the airborne ball's 3D position (review R1/R2), so the honest formulation is: ground-plane quantities (shooter position, shot location) are court-metric via the homography; near-rim event logic is **rim-normalized** — expressed relative to the annotated rim ellipse, which is the projective image of the rim circle and therefore absorbs camera geometry; reconstructed-3D quantities (T5 depth axis) are confidence-gated estimates, labeled as such.

### 5.1 Detection — theory and choice

The ball is a small object (20–40 px in 1080p wide framing). Two standard failure amplifiers: (a) feature-map stride — at stride 16, a 24 px ball is a 1.5-cell activation, so localization jitter is large relative to object size; (b) anchor/assignment starvation — few positive samples per ball instance during training. Levers, in the order we pull them: input resolution (cheapest, record-then-process makes it nearly free), P2/stride-4 head or FPN emphasis on high-res levels, tiled inference (SAHI-style) in the rim-approach window only, and fine-tuning on domain data (the data engine exists for this).

- **Stage A bootstrap:** torchvision COCO-pretrained detectors (BSD-3): `fasterrcnn_mobilenet_v3_large_fpn` (throughput) and `fasterrcnn_resnet50_fpn_v2` (accuracy reference). COCO gives `person` + `sports ball` for free; **rim is not a COCO class** — per-session manual rim ROI (one drag, static camera) covers Stage A, honestly documented as a deliberate non-problem. Why not pull in an Apache YOLO/RTMDet immediately (review R12): that adds a second training framework before the domain data that would justify fine-tuning exists; torchvision keeps Stage A's dependency surface minimal, and the ship-path detector enters at Stage B where its accuracy actually matters.
- **Ship path (Stage B):** RTMDet-tiny or YOLOX-S fine-tuned on self-collected + Roboflow CC-BY data — Apache-2.0, plain-conv ops that convert to Core ML with low risk. Configs and training scripts are Stage-A deliverables (runnable, smoke-tested on CPU at reduced scale).
- **Held option:** D-FINE-nano/RF-DETR-nano if their Core ML story resolves (tracked; Phase-0 §8).

### 5.2 Tracking — theory and choice

**Players.** Tracking-by-detection with a constant-velocity Kalman filter and Hungarian assignment on IoU. Theory: the Kalman filter is the closed-form Bayes filter under linear-Gaussian dynamics; at fixed camera and 60 fps-equivalent sampling, constant-velocity residuals are small, so IoU gating + motion prediction resolves nearly all ambiguity at 1–3 targets. We add ByteTrack's key idea — associate *low-confidence* detections to existing tracks before killing them (a half-occluded player still produces a weak box; discarding it is information loss) — and skip ReID/camera-motion compensation (no crowds, no camera motion; TrackID3x3 evidence in §0). Shooter attribution = player track with minimal wrist-to-ball distance over the 1 s pre-release window.

**Ball.** Not a SORT problem (§0). The ball alternates between ballistic flight (physics is a hard constraint) and possession (dynamics are adversarial). We run a two-mode tracker: **possession mode** — gated nearest-neighbor to the possessing player's region; **flight mode** — triggered by release detection, with a **two-level trajectory model** (review R1 — a homography cannot lift an airborne ball, and a 3D parabola does not project to an exact image parabola):
- **Level 1 — image-space quadratic** fit over short sliding windows: assumption-light and robust; used for association gating and occlusion bridging. (Corroboration: this is also what Apple's `VNDetectTrajectoriesRequest` fits.)
- **Level 2 — constrained 3D ballistic reconstruction**, used only where metric claims are made (T5, arc summaries): a parabola in a vertical plane with g fixed at 9.81 m/s², anchored by the rim's known 3D position, the release region (shooter ground position + release-height band), and the ball's known diameter (~24 cm) as a noisy per-frame depth cue filtered along the arc. Free parameters: plane azimuth + initial velocity. The fit reports residuals and a confidence that gates all downstream metric outputs.
Occlusion bridging: when detections vanish inside the rim/backboard region, the Level-1 fit *predicts* positions with widening gates; re-acquisition within the gate continues the segment, and the make/miss FSM consumes both real and predicted points with their uncertainties (A5 compares Level-1-only vs. Level-2-informed bridging). Multi-ball robustness: association gates are physics-based (a second ball rolling on the floor cannot join a flight-mode track — its candidates violate the ballistic gate), which is the principled answer to the busy-gym failure mode.

### 5.3 Lift — projective geometry, done honestly

A pinhole camera maps world plane points to image points by a homography H ∈ ℝ³ˣ³ (8 DoF): for court-plane points (z=0), x_img ≃ H · x_court. Four non-collinear correspondences determine H; we use 4–8 (corners, FT line, arc apex, center circle) solved by normalized DLT + RANSAC and refined by Levenberg–Marquardt on reprojection error — normalization (Hartley) matters because raw pixel-coordinate DLT is numerically ill-conditioned. **Error model:** ground-plane error scales ≈ h/sin²(φ) · (σ_px/f) with depression angle φ — grazing views are hyper-sensitive, elevated views are benign. We derive this analytically and validate it by Monte Carlo (A7): sample click noise σ ∈ {1,2,5,10} px, propagate to court-position error maps, overlay zone boundaries. Product output: a **camera-placement guide with error isolines** — "from a 1.5 m tripod at the sideline, corner-3 positions carry ±40 cm uncertainty; elevate to 3 m to halve it."

Subtleties handled explicitly rather than ignored: (a) **feet vs. bbox-bottom** — bbox bottom is biased by shadows and leg spread; ankle keypoints (when pose is available) with bbox-bottom fallback, and the bias is measured on marked-spot shots; (b) **the shooter is not on the court plane at release** (jump shots) — we read feet at the *last ground-contact frame before release*, not at release itself, which is both more accurate and matches basketball semantics of where the shot was "from"; (c) **rim anchoring** — the rim circle's 3D position comes from known rim height (3.048 m) + the homography-derived ground point beneath it + the annotated rim ellipse; near-rim event predicates run rim-normalized (§5.4), while the 3D anchor serves Level-2 reconstruction; (d) **"fixed" cameras drift** (review R9) — a **calibration drift monitor** tracks reprojection residuals of stable court features per minute and prompts re-calibration when residuals exceed the A7 click-noise band, converting an unstated assumption into a monitored invariant; (e) **courts without painted lines** (review R10) — the court model is configurable (NBA/FIBA/HS/custom), and a landmark-sparse mode calibrates from 4 tape-measured floor markers with zones falling back to radial distance bands from the hoop's ground projection — a supported degraded mode with wider expected T3 error (quantified by A7's noise sweep).

### 5.4 Classify — event FSM, then learned layers

**Shot FSM** (per ball-flight segment): `POSSESSION → RISING (release detected) → DESCENDING (apex passed) → RIM_INTERACTION (inside rim neighborhood) → {MADE | MISSED} → COOLDOWN`. Predicates are **rim-normalized** (review R2): the annotated rim ellipse is the projective image of the rim circle, so "inside the rim" is expressed as position relative to the ellipse (fractions of its axes), which absorbs camera geometry without requiring unobservable 3D. The MADE decision integrates the **terminal state**, not the first crossing: ball center passes downward through the rim-ellipse interior with downward image velocity, is subsequently observed (real or bridged, for N consecutive frames) in the below-net region, and does not reappear above the rim without a new possession — this is what makes rattle-in and shooter's-roll (cross, rise inside the neighborhood, drop) resolve correctly, and it is mechanically distinct from net-motion-based detection (patent posture, §10). Quantities like radius-fraction-at-crossing feed the **margin score** (distance-to-threshold ensemble: ellipse-interior margin, frames of confirmation, bridged-vs-real evidence ratio) — the raw material for probability calibration — but the verdict is the terminal state. Rebound/re-entry is guarded by COOLDOWN with hysteresis; a genuinely new flight (put-back) starts a new attempt.

**Calibration theory:** FSM margins and small-net logits are scores, not probabilities. We fit temperature scaling (single parameter, preserves ranking, minimizes NLL within its family) — and Platt scaling as the two-parameter alternative — and report reliability diagrams + ECE + Brier on *test sessions only*. Leakage discipline (review R6): val sessions are split into **val-tune** (hyperparameters, FSM grids) and **val-cal** (calibration fitting only) — tuning the FSM and then calibrating on the same sessions would double-dip and flatter the reported ECE. The split helper enforces this in code; calibration is fit once per model version, never on test.

**T4:** dribble oscillation feature (FFT energy 1–3 Hz on ball height + floor-proximity minima via homography, over the 2–4 s pre-release window with confirmed possession) → threshold rule as baseline; small temporal net (1D-CNN / shallow ST-GCN over normalized pose sequences, trained from scratch — CPU-tractable) where the rule is insufficient. **Few-shot custom types:** freeze the temporal net's penultimate embedding (or X-CLIP video embedding), classify by **prototypical head** — class prototype = mean embedding of the user's K example clips, prediction = softmax over negative distances. Theory: with K ∈ {5..40}, fitting weights overfits; metric learning sidesteps estimation by putting all capacity in the (frozen) embedding, and adding a class is a no-retraining operation — exactly the product requirement.

**T5:** at rim-arrival, decompose the rim-local miss vector into left/right (image-plane dominant — robust) and short/long (depth-dominant — azimuth-dependent). The parabola's vertical plane orientation + ball-diameter-vs-distance cue provide the depth estimate; A6 quantifies exactly how accuracy degrades as the camera approaches the shooting lane. Output includes per-axis confidence; the app hides short/long when confidence is low rather than guessing.

**T6:** vision (rim-ROI ball-position jitter — deliberately *not* net deformation, per the patent design-around, §10) + audio (log-mel window around the vision-determined rim-arrival → frozen YAMNet/BEATs embedding → logistic head). Fusion is late (average of calibrated probabilities) — the channels fail independently (occlusion vs. background noise), which is when late fusion is the right call. Design note (review R11): iPhone slow-mo retimes video but records audio at normal rate, so audio-critical sessions capture at 60 fps normal speed, and every session's opening ball-bounce doubles as an audio/video alignment check.

---

## 6. Training & Hyperparameter Tuning

- **Framework:** PyTorch (CPU in Stage A; CUDA configs ready for Stage B). Determinism: fixed seeds (torch/numpy/python), `torch.use_deterministic_algorithms(True)` where supported, seeds logged per run.
- **Experiment tracking:** **MLflow, local file store, committed under `mlruns/` export** (W&B requires an account/API key — MLflow keeps the repo self-contained and reviewer-runnable; the tracking UI is one `mlflow ui` away). Every run logs: config hash, git SHA, seed, dataset manifest hash, metrics, artifacts (plots). Reports embed figures generated from tracked runs — no hand-made numbers.
- **Config discipline:** every experiment is a YAML config under `configs/`; `python -m bball.run +experiment=<name>` reproduces it. No experiment exists unless its config is committed.
- **Augmentation (detector fine-tuning, Stage B; validated in Stage A on synthetic):** motion-blur (directional kernels matched to EDA streak statistics), color jitter + white balance shifts (ball color varies more than shape), perspective warps within plausible camera-pose deltas, copy-paste ball augmentation (paste ball crops onto court backgrounds at trajectory-consistent positions — cheap positives for a starved class), mosaic OFF for the ball class (destroys the small-object context that helps here). Each augmentation's inclusion is justified by an EDA distribution it covers; kitchen-sink augmentation is how small-object recall silently dies.
- **Tuning methodology:** stage-wise, cheap-to-expensive — (1) grid on the two or three parameters theory says dominate (e.g., FSM: rim-radius multiplier × confirmation frames N × cooldown length), (2) random search / ASHA (via Optuna) only where the grid shows sensitivity, (3) never tune on test sessions; every tuning table in the report carries its search space, budget, and val-vs-test gap so the reader can audit for overfitting-by-tuning.
- **Loss/schedule specifics (Stage B detector):** AdamW, cosine schedule with linear warmup, EMA weights; focal loss vs. CE for the ball class decided by a tracked ablation, not taste.

---

## 7. Ablation Matrix

Each ablation states its hypothesis, the metric that adjudicates, and the regime (S = synthetic, R = real footage, S→R = designed synthetic, confirmed real in Stage B). This table is the experimental core of the report.

| ID | Ablation | Hypothesis | Metric | Regime |
|---|---|---|---|---|
| A1 | Ball association, four arms: per-frame bbox + ballistic bridging **vs.** heatmap-temporal (TrackNet-lite, 3-frame) **vs.** bbox w/o bridging **vs.** background-subtraction candidates fused with bbox (review R4) | At 20–40 px, bbox+bridging ≈ heatmap on clean flight; heatmap wins only under heavy blur (30 fps) and dense occlusion; bridging closes most of that gap at far lower deploy cost; bg-sub fusion buys recall on clean flight but fails under multi-mover chaos | track completeness; T2 F1 downstream | S→R |
| A2 | Heatmap input frames ∈ {1, 3, 5} | 3 frames captures motion cue; 5 adds latency, not accuracy, at basketball scale | ball recall @ IoU 0.3 | S |
| A3 | Detector input resolution ∈ {512, 768, 1088} | Small-object recall rises steeply to 768 then saturates; resolution is the cheapest recall lever pre-fine-tuning | ball recall; mAP@0.5 | S + R (Roboflow imgs) |
| A4 | Player tracker: greedy IoU vs. Kalman+IoU vs. +low-score recovery (ByteTrack-style) | Low-score recovery is the only add-on that moves HOTA at ≤3 players; ReID unjustified | HOTA / IDF1 / ID switches | S (multi-agent sim) + TrackID3x3 if retrievable, else Stage B |
| A5 | Occlusion bridging OFF vs. Level-1 (image-space) vs. Level-2-informed (3D-anchored) × occlusion length {3, 8, 15, 30 frames} | Without bridging, T2 F1 collapses beyond ~8-frame gaps (rim-ball occlusion is longer); Level-1 bridging degrades gracefully to ~30 frames; Level-2 anchoring adds little for T2 but matters for T5 | T2 F1 vs. gap length | S |
| A6 | Camera azimuth sweep {15°, 30°, 45°, 60°, 75°, 90°} × height {1.5, 3 m} | Left/right accuracy ≈ flat across azimuth; short/long decays toward end-on views; 45–60° elevated is the knee — **this curve is the product's camera-placement guidance** | T5 per-axis accuracy | S (the sweep is exactly what synthetic is for) |
| A7 | Homography: 4 vs. 6 vs. 8 points × click noise {1, 2, 5, 10 px} × refinement ON/OFF | 6+ points with RANSAC+LM halves P90 error vs. raw 4-point DLT at realistic (2–5 px) noise; zone accuracy insensitive except in the on-line band | T3 cm error (median/P90); zone accuracy | S (Monte Carlo) + R (marked spots, Stage B) |
| A8 | FSM parameters: rim-radius multiplier × confirmation frames × cooldown | A plateau exists (robustness), not a knife-edge (overfit rule); report the sensitivity surface, not just the optimum | T2 F1 heatmap over grid | S + R events |
| A9 | Calibration: none vs. temperature vs. Platt, per data regime | Temperature scaling cuts ECE ≥ 50% with zero accuracy cost; margins are monotone-informative | ECE, Brier, reliability plots | S + R |
| A10 | T4 features: 2-feature logistic vs. pose-only temporal net vs. pose+ball-trajectory fusion | The 2-feature baseline lands ~80%+; fusion beats pose-only where the ball, not the body, carries the signal | T4 macro-F1 | R (SpaceJam proxy in Stage A) |
| A11 | Few-shot K ∈ {5, 10, 20, 40} × {NCM-DTW baseline, prototypical head} × embedding {pose-net, X-CLIP} | Accuracy-vs-K is concave with the knee at 10–20; prototypes beat NCM-DTW by a wide margin; pose embeddings suffice for kinematic classes | few-shot accuracy curve | R (Stage B; harness + SpaceJam dry run in Stage A) |
| A12 | Quantization fp16/int8 (PTQ) — **scoped to the small models we train ourselves** (TrackNet-lite, temporal head); detector PTQ + all latency numbers are Phase-2 on-device items (review R8) | ≤ 1 pt accuracy loss at int8 with per-channel PTQ; 3–4× size cut | accuracy delta, model size (CPU-only, labeled as such) | S |

Cut lines (declared now, so scope-cutting later is a decision, not a scramble): A2 and A12 are droppable without harming the core narrative; A1, A5, A6, A7, A9 are the report's spine and are not droppable.

---

## 8. Evaluation & Error Analysis

- **Stratified reporting:** every T1–T3 metric sliced by venue, camera placement, lighting, ball type, occlusion severity (from the EDA occlusion timeline), and shot zone. The cross-venue held-out set is the headline number; within-venue numbers appear alongside to expose the generalization gap explicitly (the DeepDarts lesson: 94.7 → 84.0 across viewpoints — we report our version of that gap rather than hiding it).
- **Failure galleries:** auto-generated contact sheets of the worst cases per task (missed detections in rim window, bridge failures, FSM misfires, zone misassignments on the line band) with trajectory overlays — committed to `reports/figures/`. A failure the report can show is a failure understood.
- **Error taxonomy:** each T2 error tagged {detection-miss, association-break, bridge-error, FSM-rule, label-noise} by a triage script + human pass; the taxonomy drives the next iteration's priority (fix the biggest bucket, re-run, show the delta).
- **Counterfactual audits:** replay real (Stage B) sessions with single components ablated (e.g., perfect-ball-track oracle from labels) to attribute end-to-end error to stages — the honest way to answer "is the detector or the FSM the bottleneck?"
- **Calibration:** reliability diagrams per data regime; a model that is accurate but miscalibrated fails the gate (§1.3) because session aggregates are the product.
- **Statistical hygiene:** per-session bootstrap CIs on headline metrics (sessions are the exchangeable unit); no claim of improvement without non-overlapping CIs or a paired per-session test.

---

## 9. Checkpoint & Phase-2 Outline

### 9.1 Phase-1 checkpoint (the gate)

Phase 1 is DONE when: (1) Stage-A gates in §1.3 hit on synthetic + available real data; (2) the non-droppable ablations (A1, A5, A6, A7, A9) are run, tracked, and written up; (3) the pipeline runs end-to-end on a real clip via one command and the demo notebook; (4) per-stage reports + final report exist with figures from tracked runs; (5) the Stage-B runbook (data protocol, training configs, eval commands) is executable by the user without code archaeology. **The checkpoint review presents:** headline metrics table, the A6 azimuth curve, the calibration plots, the failure gallery, and the go/no-go recommendation per tier for Stage B.

### 9.2 Phase 2 — deployment outline (definite next phase; depth deferred)

- **Architecture:** on-device record-then-process. Capture at 1080p60 (240 fps burst only in rim-approach if needed — thermal + storage tradeoff); processing pass on-device via Core ML (small CNN detector INT8 on ANE, ~4–10 ms/frame at decimated rate + full-rate rim windows) with the same Python-validated logic ported to Swift; coarse live UI (attempt counter) via Vision primitives at 30 fps if desired.
- **Ship path v0 (zero third-party weights):** `VNDetectTrajectoriesRequest` + native body pose + tap-calibration homography + FSM port — validates UX and the patent design-around posture while the trained detector matures.
- **Model pipeline:** PyTorch → coremltools 9 → fp16 → int8 PTQ (per-channel) → on-device eval harness replaying Stage-B clips; distillation only if PTQ misses the accuracy gate.
- **Latency/energy budget (to validate):** 30 s session clip processed ≤ 30 s on A17-class ANE; ≤ 5% battery per 30 min session.
- **Open questions (resolved post-checkpoint):** VNDetectTrajectories behavior at slow-mo timebases; Vision pose adequacy vs. RTMPose conversion; DETR-family Core ML status re-check (RF-DETR #318); audio capture pipeline (AVAudioSession) alignment with video timestamps at slow-mo; Swift FSM port test strategy (golden-file parity against Python on the same clips).

---

## 10. Limitations & Ethics

- **Patent exposure:** US11594029 (feet-at-release→homography location) and US11380100 (net-motion-based result detection) read closely on parts of this design at abstract level. R&D proceeds; commercialization requires a professional FTO opinion. Design-around notes are maintained from day one (e.g., our result logic never inspects net motion; alternatives to foot-projection location are documented as they arise). This plan is not legal advice.
- **Single-camera physics:** short/long miss direction near the shooting lane is structurally under-observed; we quantify (A6) and confidence-gate rather than promise. Swish-vs-rattle from vision alone at wide framing may not clear usefulness; the audio channel is the designed escape hatch, and "experimental, off by default" is the honest shipping mode for T6.
- **Sim-to-real:** synthetic experiments validate geometry, logic, and robustness *mechanisms*; they cannot certify real-world perception accuracy. Every synthetic number is labeled as such; Stage B exists precisely to close this loop, and the sim-vs-real EDA comparison (§3) audits simulator fidelity.
- **Domain shift:** trained on one user's venues/balls; generalization beyond the collection grid (night outdoor courts, double rims, netless rims) is unverified until tested. Netless rims specifically weaken two cues (net occlusion timing, net deformation) — flagged for a dedicated Stage-B session.
- **Privacy:** filming public courts captures bystanders. Mitigations: on-device processing as an architecture-level privacy stance (footage never leaves the phone), face-region blurring in any shared/exported clips or published figures, consent for any non-user player who is identifiable, and no cloud retention. Self-collected dataset is not redistributed with identifiable third parties.
- **License compliance:** the ledger (`LICENSES.md`) is a maintained artifact; weights-vs-code license splits (VideoMAE-style traps) are re-checked at every dependency addition. Nothing AGPL/CC-NC/unlicensed ships.
- **Honesty boundary:** self-reported accuracies from hobby repos (88–97%) are treated as unaudited folklore; our targets are set from first principles and DeepDarts' honest cross-condition numbers, and our own headline number is always the cross-venue one.

---

## 11. Execution Plan — Staging, Repository, Reproducibility

### 11.1 Two-stage reality (compute honesty)

**Stage A — this repository, CPU-only container (now):** full `bball` package (detect/track/lift/classify), synthetic trajectory+rendering engine grounded in measured court/ball physics, all geometry and event logic with unit tests, EDA tooling, evaluation harness with all metrics, MLflow tracking, ablations A1–A9 in their synthetic regimes (+ real-image A3 if Roboflow sets are retrievable), small-scale trainings (TrackNet-lite, temporal heads) at CPU-feasible scale, per-stage reports, demo notebook. **Stage A makes no real-world accuracy claims** — it delivers validated machinery, tuned logic, quantified geometry, and the exact harness Stage B fills with real numbers.
**Stage B — user GPU + self-collected footage (next):** execute `docs/DATA_PROTOCOL.md`, run the data engine to 5–10 k labeled frames, fine-tune the ship-path detector, re-run the full ablation matrix's R-regimes with the *same commands and configs*, produce the cross-venue headline numbers, then the Phase-1 checkpoint review (§9.1).

Gates for Stage A (the build's acceptance criteria): **G1** end-to-end run on a provided clip (real if retrievable, else synthetic-rendered video) via one command producing the session report; **G2** unit tests green on geometry/FSM/bridging (including rebound double-count and multi-ball association cases); **G3** non-droppable ablations tracked in MLflow with committed figures; **G4** sim parameter ranges traceable to measured/cited physics (no fantasy constants); **G5** every report figure regenerable from a committed config + seed.

### 11.2 Repository structure

```
├── README.md                  # project story, headline results table (regime-labeled), quickstart
├── pyproject.toml             # pinned deps; CPU-only install path
├── Makefile                   # make setup / test / demo / eda / ablations / reports
├── configs/                   # YAML per experiment; hydra-style composition
├── src/bball/
│   ├── detect/                # torchvision bootstrap, TrackNet-lite, detector interfaces
│   ├── track/                 # kalman.py, association.py, ballistic.py (bridging)
│   ├── lift/                  # homography.py (DLT+RANSAC+LM), court_model.py, rim_frame.py
│   ├── events/                # fsm.py, release.py, miss_direction.py, calibration.py
│   ├── heads/                 # shot_type.py, fewshot.py, audio.py
│   ├── synth/                 # physics.py, camera.py, render.py, scenarios.py
│   ├── eval/                  # metrics.py, stratify.py, galleries.py, bootstrap.py
│   └── viz/                   # overlays, court plots, reliability diagrams
├── data/                      # gitignored; manifests + LICENSES.md committed
├── notebooks/demo.ipynb       # clip → detect → track → lift → classify → visuals
├── reports/                   # phase0_research.md, per-stage reports, figures/
├── tests/                     # pytest: geometry, FSM, bridging, splits (leakage guard)
└── mlruns-export/             # tracked-run summaries backing report figures
```

### 11.3 Reproducibility contract

Pinned `pyproject.toml` (+ lock), fixed seeds logged per run, dataset manifests with hashes, configs-as-experiments, MLflow run IDs cited in reports next to every figure, CI-light (`make test` + a 60 s smoke pipeline on a bundled mini-clip), and a `REPRODUCING.md` that maps every report figure to its command.

### 11.4 Risk register (top 5, with mitigations)

| Risk | Exposure | Mitigation |
|---|---|---|
| Public sets unreachable from container (Roboflow key, Drive rate limits) | A3/A4 real regimes slip | Synthetic regimes proceed; downloader scripts + instructions ship for Stage B; decision documented per dataset |
| COCO `sports ball` recall too low near rim even at high res | Stage-A real-clip demo weakens | Rim-window tiling (SAHI-style), temporal prior gating, and honest reporting; fine-tuning is Stage B's first job |
| CPU budget starves trainings (TrackNet-lite, temporal heads) | A1/A10 underpowered | Reduced resolution/frames documented as such; configs scale up unchanged on GPU; claims scoped to the regime run |
| Synthetic engine fidelity questioned | Ablation credibility | Ground every parameter in cited physics/EDA measurements; publish the sim-vs-real audit when Stage B data lands |
| Patent posture drifts as features grow | Legal | Design-around log reviewed at each feature addition; FTO before commercialization |

---

*Plan v1.0 was drafted from the Phase-0 survey; v1.1 incorporates the adversarial review in `docs/PLAN_REVIEW.md` (12 findings, all resolved or explicitly accepted as risks).*
