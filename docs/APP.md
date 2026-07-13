# The App — Web Workbench Today, Native iOS in Phase 2

## What exists now (built, tested, in this repo)

A local web app (`make app` → http://localhost:8000) that is both the **Stage-B labeling
workbench** and the **product-experience preview**, reusing the pipeline directly:

| Tab | What it does |
|---|---|
| **Session** | Open a video by path or upload; per-session state lives under `data/app_sessions/<sid>/` |
| **Calibrate** | Scrub to a clean frame, tap the named court landmarks (≥4) → homography with RMS report, and **the projected 3PT/paint lines are drawn back onto the frame** — if the green lines hug the paint, the calibration is right. Switch to rim mode, tap ≥5 points on the rim → fitted rim ellipse overlay |
| **Review** | Run analysis (bg-sub spine; zero weights needed) → proposed shots with timestamps; video player seeks to each event; correct outcome / miss direction / type / quality; **add-missed-shot** at the current playhead (honest false-negative accounting); exclude non-attempts; save `labels.csv` (same schema as `scripts/review_events.py`) |
| **Zones** | Presets (basic3 / extended / spots) with live parametric controls (interior radius, deep-three offset, sector angles), or **draw custom polygons on the top-down court** (click to add vertices, double-click to close and name); apply to the session — if calibrated, zone boundaries are also **projected into the camera frame** |
| **Results** | Shot chart on the court (makes green / misses red, on-the-line ringed), per-zone attempts/makes/FG% under the active partition — re-buckets instantly when the partition changes |

**Phone usage today (the honest "mobile version"):** run `make app` on your computer, open
`http://<computer-ip>:8000` from the phone on the same Wi-Fi. The layout is responsive and
the page is a PWA (add to home screen). Record with the native Camera app per the protocol;
label on the couch from either device.

Analyzer note: the built-in proposal engine is the classical bg-sub spine (works with zero
downloaded weights, tuned for fixed cameras). When torchvision COCO weights are available it
also lifts the shooter's feet through the homography at each release for shot location;
without them, locations stay blank and are filled by Stage-B's fine-tuned detector. Proposal
quality on real footage improves exactly as Stage B trains — the workbench is how those
labels get made.

## What is deliberately NOT built yet (and why)

**Native iOS (Swift + Core ML, on-device processing)** is Phase 2, gated behind the Stage-B
checkpoint (plan §9.2), and honestly cannot be built or tested from this Linux container —
it needs Xcode/macOS. The workbench de-risks it anyway: every UI flow here (tap-calibration,
review-correct loop, zone drawing, shot chart) is the blueprint for the native screens, and
the pipeline logic it exercises is the same logic the Swift port will golden-file against.
Deployment path and open questions: `docs/PROJECT_PLAN.md` §9.2 (Core ML conversion,
fp16/int8 PTQ, `VNDetectTrajectoriesRequest` scaffold, latency budget).

## Run it

```bash
make setup          # once (installs app extras too)
make app            # serve on 0.0.0.0:8000
```

State layout per session: `state.json` (video path, homography, rim, active partition),
`labels.csv` (the ground truth you create), uploaded video (if uploaded rather than pathed).
