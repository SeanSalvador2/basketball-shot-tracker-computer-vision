# Phase 1 — Pipeline Report: Architecture as Built (Stage A)

What was actually implemented, where it deviates from `docs/PROJECT_PLAN.md` v1.1, and why.
Companion reports: `phase1_eda.md` (data analyses), `phase1_experiments.md` (ablations),
`phase1_summary.md` (gate status). Regime for every number here: **S (synthetic)**.

## 1. The spine as built

```
clip.mp4 ──► [DETECT] per-frame ball candidates          src/bball/detect/
             • bgsub.py       MOG2 + shape/temporal filters (classical, zero weights)
             • torchvision_detector.py  Faster R-CNN wrapper (weight-agnostic; see D6)
             • tracknet_lite.py         3-frame heatmap UNet (A1/A2 arm, CPU-trainable)
             • interfaces.py            one BallCandidate/Detection contract for all sources
         ──► [TRACK] association + bridging              src/bball/track/
             • kalman.py      own CV Kalman (players); association.py Hungarian+IoU
                              + ByteTrack-style low-score second pass
             • ballistic.py   two-level trajectory: L1 image-space quadratic (gating +
                              gap fill, widening gates, stale-model reseed) and L2
                              gravity-constrained vertical-plane parabola anchored by rim
                              3D + release region + ball-diameter cue (confidence-gated)
         ──► [LIFT] one-time session calibration          src/bball/lift/
             • homography.py  own normalized DLT + RANSAC + LM (cv2 only as test oracle)
             • court_model.py NBA/FIBA/HS/custom dims, zones, on-line band, radial mode
             • rim_frame.py   rim ellipse annotation + rim-normalized coords + rim 3D anchor
             • projection.py  pinhole camera + the h/sin²(φ) error model
         ──► [CLASSIFY] events                            src/bball/events/
             • release.py     release detection (pose + fallback), last-ground-contact,
                              flight segmentation
             • fsm.py         rim-normalized FSM, terminal-state MADE, margin score
             • miss_direction.py  per-axis calls with confidence gating
             • calibration.py temperature + Platt + reliability/ECE/Brier
```

Supporting layers: `synth/` (ballistic generator with cited constants, scene camera,
procedural renderer + detection-noise model, session/negative generators), `eval/`
(tolerance-matched event PRF, FP/hour, T3 cm, zone confusion, per-axis T5, leakage-guarded
splits with val-tune/val-cal, per-session bootstrap, failure galleries), `viz/`, `ablations/`
(A1, A5–A9 with MLflow tracking + committed exports), `pipeline.py`/`demo.py` (end-to-end
orchestration), 100 unit tests.

## 2. Design decisions that survived contact with implementation

- **Rim-normalized FSM, sharpened.** The plan's predicates held, but implementation forced
  precision: the make gate is the **lateral offset along the rim-ellipse major axis at the
  descending rim-level crossing, interpolated to exactly rim level** (the ball moves fast
  enough near the rim that the first post-crossing frame under-reads the offset — this
  interpolation was the difference between 9/10 and 10/10 on the scripted scenarios). The
  minor axis is deliberately **not** a gate: near-rim-height cameras image the rim almost
  edge-on (EDA §4), making it an artifact-prone signal. Depth misses are still rejected
  off-axis because the depth offset projects onto the major axis at any azimuth off the
  shooting lane — which is also exactly why end-on cameras lose short/long (A6).
- **Terminal state = net-dwell with ordered pop-out.** MADE requires a net-region dwell of
  `confirm_frames` after the crossing, with a rise back above rim level *after settling*
  scored as a rim-out, while a rise *before* settling is tolerated (shooter's roll). Frames
  lost to occlusion below the rim **count as make evidence** — a ball that crosses down
  through the interior and vanishes into the net is the make signature, not missing data.
- **Two-level trajectory, with a sharper division of labor than planned.** A5 measured that
  L2 (global gravity-constrained parabola) is a *worse gap-filler* than the local L1
  quadratic (model bias), so: gating and filling are always L1; L2 exists solely for metric
  outputs (T5, arc summaries) behind its confidence. Review R1's instinct — each level only
  for what it can support — turned out to be the empirically correct division too.
- **Gate hygiene beats gate strictness.** The L1 association gate initially rejected the
  possession→flight transition (a stale static-window fit outruled the launching ball) and
  collapsed whole tracks. Fix: two consecutive gate rejections ⇒ the model is stale, reseed
  from data. Isolated outliers (a second ball) still bounce off the gate — the multi-ball
  scenario test covers both sides.

## 3. Deviation log (D1–D10)

| # | Deviation | Why / consequence |
|---|---|---|
| D1 | **torch install path.** Plan: CPU wheels from `download.pytorch.org/whl/cpu`. That host (and its `download-r2` mirror) returns 403 through this container's egress proxy — an org policy denial, not retried. Fallback: PyPI `torch==2.8.0` (CUDA build) + the `nvidia-*-cu12` runtime libs it hard-imports. | Execution is CPU-only (`cuda_available=False`); the NVIDIA libs are dead weight in-container only. `make setup` on any normal machine uses the clean CPU index. Ledger note in `LICENSES.md`. |
| D2 | **COCO-pretrained weights not fetchable** (same blocked host). | `TorchvisionBallPlayerDetector` is weight-agnostic: tries pretrained, falls back to random-init and flags it; plumbing (resolution control, class mapping) is smoke-tested. Stage-A detection is carried by bg-sub + the synthetic noise model; A3 moves to Stage B. |
| D3 | **MLflow 3.14 gates the file store** behind `MLFLOW_ALLOW_FILE_STORE=true` (maintenance mode). | Flag set inside `ablations/common.py`; the self-contained file store (plan's R12 rationale) is preserved. If MLflow removes it, `sqlite:///mlflow.db` is a one-line change. |
| D4 | **`detect/interfaces.py` written in M3** (plan: M4). | The synthetic detection-noise model emits the same `BallCandidate` the detectors do — defining the contract first is what lets logic ablations run without any detector. |
| D5 | **`release_t` = first flight frame** (was: last possession frame). | The off-by-one poisoned L2 fits anchored at the release point (16 px reprojection RMS from a single non-ballistic frame). Matches the plan's release definition semantics. |
| D6 | **Close shots steepen** in the generator: launch angle is clamped up to the minimum feasible angle for the target (a parabola cannot pass through a point above its launch ray), with a 0.25 m minimum horizontal travel guard. | Every scripted location is realizable; short-range arcs are steep, as in reality. Angles beyond the cited 45–55° band appear only for these close shots and are visible in the EDA angle histogram. |
| D7 | **A6 realism injection** (2 px jitter, 15% radius noise, 0.25 m feet-anchor noise). | With noise-free synthetic observations the rim-anchored L2 resolves *every* axis at *every* azimuth — a perfectly clean simulator makes the sweep vacuous. The injected magnitudes are the measured/typical Stage-A values (detector jitter from the noise model; feet error from A7's own T3 band). |
| D8 | **A7 sweeps DLT vs DLT+LM without RANSAC.** | RANSAC with a tight inlier gate on *gaussian* click noise discards valid points and blew P90 up 4–17×; it exists for gross outliers and stays in the production estimator + its own unit test. |
| D9 | **MOG2 shadow detection off by default.** | MOG2 labels darker-than-background movers "shadow" (127); a ball on a bright floor is exactly that, and the shadow filter silently deleted it. |
| D10 | **Demo clip is 0.06 MB** (vs the plan's 10–20 MB allowance) at 0.4× render scale, 6 shots. | Smaller is better for a committed asset; dimensions chosen divisible by the codec macro-block so no resize occurs. |

## 4. Data downloads (time-boxed attempt, plan's risk register realized)

| Source | Outcome | Artifact |
|---|---|---|
| Roboflow ball/rim sets | **Blocked** — `api.roboflow.com` / `app.roboflow.com` 403 at the egress proxy (org policy; not retried) and would need an API key regardless | `scripts/data/roboflow_download.py` (parameterized, `ROBOFLOW_API_KEY` env, per-dataset license notes) for Stage B |
| TrackID3x3 | **Partial success** — GitHub reachable: repo + **CC BY 4.0 annotation set fetched via git (~377 MB)** under `data/external/` (gitignored); the video payloads live on Google Drive, which is proxy-blocked (gdown 403) | `scripts/data/trackid3x3_download.py` with the Drive file IDs for Stage B |
| Hugging Face hub | **Success** — license-gated fetches: `emirsahin/basketball-ball` (MIT, ~215 MB ball-detection images) and a 2-clip sample of `ZhiChengAI/Basketball_V0` (CC BY 4.0, ~17 MB real gameplay clips); NC/ND/unlicensed candidates inspected and refused | `scripts/data/hf_download.py` (refuses non-MIT/CC-BY licenses without `--force`) |

Real data is a bonus, not a dependency: nothing above is load-bearing for Stage A. The two
real HF clips lack fixed-camera calibration (no rim annotation, unknown camera), so they are
Stage-B sanity-check material, not demo material — the demo states this plainly.

## 5. What Stage B inherits

Runnable, tested machinery: the full package + 100 tests + `make test/demo/eda/ablations`;
the collection protocol (`docs/DATA_PROTOCOL.md`) with the metadata sheet and negative-block
design; downloader scripts with license gates; leakage-guarded splits with val-tune/val-cal;
an FSM whose parameter surface is a plateau (A8) and a calibrator choice (Platt) already
validated in the S regime; and the exact commands (`docs/REPRODUCING.md`) whose re-run on
real footage produces the R-regime table the README will ultimately report.
