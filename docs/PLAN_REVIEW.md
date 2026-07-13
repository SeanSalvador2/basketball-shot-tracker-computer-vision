# Plan Review — Adversarial Critique & Revision Log

**Scope:** self-review of `docs/PROJECT_PLAN.md` v1.0 before implementation kickoff. Method: re-read the plan as a hostile reviewer looking for (a) correctness errors in the theory, (b) unjustified choices, (c) missing baselines, (d) leakage/overclaim risks, (e) compute-reality mismatches. Findings below; each carries a resolution status. Fixes are applied in plan v1.1.

---

## R1 — CORRECTNESS: the ball's 3D position is not observable from the homography (v1.0 §5.2)

**Finding.** v1.0 said the flight-mode parabola is fit "in court-plane coordinates: x(t) linear, z(t) quadratic." That is not a valid operation: a homography lifts **ground-plane** points only. An airborne ball's image position does not determine its 3D position from one camera; treating lifted coordinates as 3D ball positions would silently produce garbage metric outputs. Worse, the projection of a 3D parabola under a pinhole camera is not exactly a parabola in image space either (it is only near-parabolic for modest fields of view and flight arcs).

**Resolution (applied).** Two-level trajectory model, each level used only for what it can support:
- **Level 1 — image-space quadratic** over short temporal windows: assumption-light, robust, used for association gating and occlusion bridging (this is also what Apple's `VNDetectTrajectoriesRequest` does, which corroborates the engineering choice).
- **Level 2 — constrained 3D ballistic reconstruction** used only where metric claims are made (T5 miss direction, release/arc summaries): a parabola in a vertical plane with gravity fixed at g = 9.81 m/s², anchored by (i) the rim's known 3D position, (ii) the release region (shooter ground position + release height band), and (iii) the ball's known diameter as a noisy per-frame depth cue filtered along the arc. The plane orientation and initial velocity are the only free parameters; the fit reports residuals and a confidence that gates T5 output.
The A5 ablation now also compares Level-1-only bridging vs. Level-2-informed bridging.

## R2 — CORRECTNESS: make/miss predicates as written required unobservable 3D ball positions (v1.0 §5.4)

**Finding.** The FSM's MADE criterion ("ball center crosses the rim plane top-to-bottom within 0.9× rim radius of rim center in the rim-local frame") quietly assumes per-frame 3D ball positions near the rim — the same unobservability as R1. A reviewer would (correctly) reject the "all logic in metric coordinates, never pixels" invariant as stated, because it is not implementable for the airborne ball.

**Resolution (applied).** The decision logic is restated in **rim-normalized image coordinates**: the annotated rim ellipse *is* the projective image of the rim circle, so predicates are expressed as fractions of the rim ellipse axes and positions relative to it (e.g., "ball center passes downward through the rim-ellipse interior and is next observed in the below-net region without reappearing above"). This is placement-transferable — normalization by the rim ellipse absorbs the camera geometry — without requiring unobservable 3D. Level-2 3D estimates (R1) contribute *features and confidences*, not the verdict. The design invariant is reworded from "all logic in metric coordinates" to "no raw-pixel thresholds: all rules in rim-normalized or court-metric units, whichever is actually observable."

## R3 — CORRECTNESS: the shot-attempt definition excluded airballs (v1.0 §1.1)

**Finding.** A shot attempt was defined as a flight whose trajectory "enters the rim neighborhood." An airball never enters the rim neighborhood, so the definition would not count it — yet an airball is unambiguously a shot attempt and a miss (and typically a "short" or "long" one for T5). Event counts and miss-direction stats would be silently biased optimistic.

**Resolution (applied).** Redefined: a shot attempt is a released ball flight with apex above rim height whose horizontal terminal direction is toward the hoop (terminal ground-projected distance to hoop decreasing and endpoint within a generous hoop-centered radius, OR backboard contact). Airballs qualify; passes and dribbles still do not (lob passes toward the rim region are the known hard negative — measured explicitly on the negative-block data introduced in R7). Put-backs/tip-ins count as new attempts; the COOLDOWN guard prevents double-counting a single flight, not genuinely new attempts.

## R4 — MISSING BASELINE: background subtraction on a fixed camera

**Finding.** The camera is *fixed* — the single strongest classical prior available — and v1.0's baseline table never used it. Frame-differencing / MOG2 background subtraction is the canonical cheap ball-candidate generator on static cameras, likely strong at wide framing where the moving ball is the dominant small mover. Without it, "fine-tuned deep detector beats X" has a hole where the obvious classical X should be.

**Resolution (applied).** Added as (a) a ball-detection baseline (MOG2 + size/shape filtering + temporal consistency), and (b) a candidate-proposal channel that can be fused with the neural detector (union of candidates, detector scores arbitrate) — potentially rescuing small-ball recall cheaply. A1 gains this third arm. Hypothesis registered: background subtraction has high recall on clean flight but fails under occlusion and multi-mover chaos; fusion inherits the recall without the failures.

## R5 — OVERCLAIM RISK: "court-plane coordinates" language would have leaked into reports

**Finding.** Consequence of R1/R2: several report-facing phrases ("metric trajectory," "3D rim-local frame") promised more observability than a single camera has. If the reports repeat them, a knowledgeable reviewer discounts the whole project.

**Resolution (applied).** Terminology pass: "court-metric" reserved for ground-plane quantities (shot location, shooter position); "rim-normalized" for near-rim image-space logic; "reconstructed 3D (confidence-gated)" for Level-2 outputs. The evaluation section already reports per-axis T5 accuracy; now the *language* matches the physics everywhere.

## R6 — LEAKAGE: calibration and FSM tuning share validation data with model selection

**Finding.** Temperature scaling is fit on val sessions (§6), and FSM parameters are tuned on val sessions (A8). Selecting FSM parameters and then calibrating on the same sessions double-dips: the calibrated probabilities inherit selection bias, and reported val ECE flatters. Test-set discipline alone doesn't fix in-val double-dipping.

**Resolution (applied).** Val is split into `val-tune` (hyperparameters, FSM grid) and `val-cal` (calibration fitting only), by session as always. ECE/Brier/reliability are *reported* only on test sessions. Documented in §6; the split helper enforces it in code, and the leakage-guard unit test covers it.

## R7 — MISSING DATA: no pure-negative footage in the collection protocol

**Finding.** T1 precision needs footage where *no shots occur* (dribbling drills, passing, layup lines with the ball rolling around) to measure false-attempt rate honestly. v1.0's protocol had "free play" but no guaranteed shot-free segments, so FP rate would be measured on shot-dense footage only — an easy grader.

**Resolution (applied).** Protocol adds ≥ 5 min per session of explicit negative blocks (ball activity, zero shot attempts) + a lob-pass block (the adversarial near-positive for the R3 definition). T1 metrics now report FP/hour-of-negative-footage alongside event F1.

## R8 — COMPUTE HONESTY: A12 quantization ablation was mis-scoped for Stage A

**Finding.** Post-training int8 quantization of torchvision detection models on CPU is fiddly and low-value (they are not the ship model); promising "quantization impact on accuracy vs latency" in Stage A overstates what a CPU container can honestly measure (no ANE latency exists off-device).

**Resolution (applied).** A12 rescoped: Stage A quantizes only the small models we train ourselves (TrackNet-lite, temporal head) and reports accuracy/size deltas, explicitly labeled CPU-only; detector PTQ and all latency numbers move to Phase 2 on-device harness. A12 remains on the cut line.

## R9 — ROBUSTNESS GAP: "fixed camera" is not perfectly fixed

**Finding.** Tripods on fences/bleachers drift with wind and bumps. A one-time homography silently degrades; T3 error would grow mid-session with no alarm, and the "fixed camera" premise would fail in exactly the field conditions the product targets.

**Resolution (applied).** Added a **calibration drift monitor**: track reprojection residuals of stable court features (line intersections via lightweight template/feature tracking) per minute; alert + prompt re-tap (or auto re-estimate from tracked features) when residuals exceed the click-noise band from A7. Doubles as a data-quality flag in session metadata. Cheap, and it converts an unstated assumption into a monitored invariant.

## R10 — SCOPE GAP: courts without painted lines (driveways) break both calibration and zones

**Finding.** The generalization story ("driveway hoops") conflicted with a calibration flow that assumes visible court landmarks and zone definitions that assume painted 3PT lines.

**Resolution (applied).** Court model is configurable (NBA/FIBA/HS/custom); calibration supports landmark-sparse mode — user places 4 markers at tape-measured distances (the protocol already includes a tape measure) and zones fall back to radial distance bands from the hoop's ground projection. Documented as a supported degraded mode with wider expected T3 error (quantified by A7's noise sweep).

## R11 — MISSING RISK: slow-mo audio/video alignment for T6

**Finding.** iPhone slow-motion video retimes video but records audio at normal rate; naive frame-timestamp alignment of the audio window around rim-arrival can be off by enough to clip the transient. T6's audio channel depends on this alignment, and v1.0 only flagged it in Phase-2 open questions.

**Resolution (applied).** Promoted to a T6 design note + Stage-B protocol item: prefer 60 fps normal-speed capture for audio-critical sessions, and validate slow-mo audio track alignment empirically (impulse test: bounce the ball on camera at session start — the bounce is a natural clapperboard). Phase-2 question retained for the AVFoundation specifics.

## R12 — JUSTIFICATION GAP: MLflow-over-W&B and torchvision-bootstrap choices lacked stated alternatives

**Finding.** Two engineering choices were asserted rather than argued: experiment tracking (MLflow vs. W&B) and the Stage-A bootstrap detector (torchvision vs. pulling an Apache YOLO immediately). Reviewers flag unargued tool choices in a plan that argues everything else.

**Resolution (applied).** One-line justifications added where the choices appear: MLflow file-store keeps the repo self-contained and runnable by a reviewer with zero accounts/keys (W&B's collaboration features buy nothing for a solo reproducible portfolio); torchvision bootstrap avoids adding a second training framework before the data that would justify it exists (COCO pretraining covers `person`+`sports ball`; the ship-path detector enters with Stage-B fine-tuning where its accuracy actually matters).

---

## Accepted limitations (reviewed, deliberately not changed)

- **No calendar/effort estimates.** Staging + gates order the work; time estimates would be theater from inside a container.
- **A4 (tracker ablation) may lack real multi-player data in Stage A.** Synthetic multi-agent + TrackID3x3-if-retrievable, else the harness ships and Stage B fills it. Stated in the matrix.
- **T6 remains experimental with no accuracy commitment.** That is the honest posture given zero prior art; the audio angle is the designed upside.
- **Patent posture is documentation + design-arounds, not avoidance.** R&D is lawful; the FTO gate sits before commercialization, and that is where it belongs.

**Verdict:** v1.0 was structurally sound but contained two real correctness errors (R1, R2) that would have propagated into code and reports, one definitional bug (R3), and one missing classical baseline (R4) — exactly the failure classes adversarial review exists to catch. v1.1 is approved for implementation kickoff.
