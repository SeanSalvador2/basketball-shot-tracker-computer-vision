# Phase 1 — Stage A Summary & Gate Status

Stage A (CPU container, synthetic regime) is complete. This page is the honest ledger:
what passed its gates, what is validated, what remains unproven until Stage B's real
footage. **Every number below is regime S (synthetic)** unless explicitly marked otherwise.

## Stage-A acceptance gates (plan §11.1)

| Gate | Requirement | Status | Evidence |
|---|---|---|---|
| G1 | End-to-end run on a clip via one command producing the session report | **PASS** (synthetic clip; no calibrated real clip was obtainable in-container) | `make demo` executes `notebooks/demo.ipynb` top-to-bottom: bundled clip → bg-sub detection → bridged track → FSM → shot chart + session summary |
| G2 | Unit tests green on geometry/FSM/bridging incl. rebound double-count and multi-ball | **PASS** | 100/100 tests (incl. 11 heads-scaffold harness tests); scripted scenarios: clean make, 4 miss directions, rattle-in, shooter's roll, rim-out, put-back (two attempts, no double count), lob-pass negative, multi-ball distractor, occluded make |
| G3 | Non-droppable ablations (A1, A5, A6, A7, A9) tracked in MLflow with committed figures | **PASS** (+ A8) | `mlruns-export/*.csv|json` (run IDs inside), `reports/figures/ablations/*.png`, configs in `configs/ablations/` |
| G4 | Sim parameters traceable to cited physics | **PASS** | Constants cited in `synth/physics.py` (g, rim 3.048 m/0.4572 m, ball 0.24 m, release 6–9 m/s / 45–55°); EDA trajectory audit confirms apex/flight-time plausibility; deviations (close-shot steepening) documented |
| G5 | Every report figure regenerable from committed config + seed | **PASS** | `docs/REPRODUCING.md` maps each figure → command; seeds fixed in configs |

## What works (validated in regime S) vs what is unproven

| Claim | Status | Where shown |
|---|---|---|
| Homography calibration: ≥6 points ⇒ ≤5 cm median / 11 cm P90 at 2 px click noise; zone acc ≥0.99 | **Validated (S)** — meets the ≤10 cm Stage-A gate | A7 |
| Camera-placement guidance: per-axis T5 accuracy vs azimuth, axes trade as the camera rotates; 45–60° balances; confidence gate hides unreliable short/long | **Validated (S)** — the deliverable IS the curve | A6 |
| Rim-normalized FSM: 0.86–0.98 make/miss F1 across a 36-cell parameter plateau; terminal-state logic resolves rattle-in / roll / rim-out / put-back correctly | **Validated (S)** | A8 + scenario tests |
| L1 ballistic bridging: F1 0.96→0.72 over 3→30-frame rim occlusions; necessary for track completeness (0.53→0.86) | **Validated (S)** | A5, A1 |
| Platt calibration cuts ECE 77% on a held-out venue (0.117→0.027); temperature does NOT fit FSM margins | **Validated (S)** — negative result kept | A9 |
| Leakage discipline: session/scene splits, held-out venue, val-tune/val-cal separation | **Validated (in code)** — enforced by tests | `eval/splits.py` |
| Real-footage detection recall (ball at 20–40 px near the rim) | **Preliminary (R, zero-shot)** — COCO weights *were* fetchable (block was transient); zero-shot on real emir-shoots frames the ball fires in ~35% (mobilenet) / ~75% (resnet50) of frames, people in 95–100%. No GT boxes ⇒ fire-rate, not IoU recall; no fine-tune yet | A3 (real panel); Stage B fine-tune |
| Real make/miss accuracy ≥92% F1 (Stage-B target) | **Unproven** — synthetic F1 is by construction easier | Stage B |
| T3 cm error on tape-measured real spots | **Unproven** (S bound: 5–10 cm at careful clicks) | Stage B |
| Heatmap-vs-bbox decision at real basketball scale | **Open** — A1's reduced-scale S run cannot settle it | Stage B |
| T4 shot type / T6 audio | **Scaffold only** — harness-validated, no accuracy claims | `bball.heads`, Stage B |
| Calibration drift monitor (R9) | **Designed, not implemented** — needs real footage with drift to be testable | Stage B |

## Headline Stage-A numbers (all S)

- FSM batch make/miss accuracy on ground-truth tracks: **89–96%** across 4 camera
  placements (best at 60° azimuth), with attempt-recall misses concentrated at point-blank
  range.
- Demo end-to-end (bg-sub detection, no weights): **5/6 shots correct**; the one failure is
  a T1 visibility case (ball hidden behind the backboard at point-blank range), narrated in
  the notebook.
- A6, A7, A5, A9 tables: see `phase1_experiments.md`.

## Stage-B runbook (pointer)

1. **Collect** per `docs/DATA_PROTOCOL.md` (variation grid, negative blocks, marked spots,
   metadata sheets). Hold out one venue entirely.
2. **Fetch** public seeds: `scripts/data/roboflow_download.py` (needs API key),
   `trackid3x3_download.py` (Drive IDs recorded), `hf_download.py` (license-gated).
3. **Install** on the GPU box: `make setup` (clean CPU/CUDA torch path), `make test`.
4. **Label** via the semi-automatic loop (pipeline proposes, human corrects; CVAT + SAM for
   boxes; rim ellipse once per session).
5. **Re-run the R regimes with the same commands**: `make eda` (sim-vs-real fidelity audit),
   `make ablations` (A1/A3/A5/A8/A9 on real sessions; A6's real analog = the 5-placement
   collection sweep), detector fine-tune (RTMDet-tiny/YOLOX-S per plan §5.1).
6. **Report**: fill the R column of the README results table; run the §9.1 checkpoint
   review (headline metrics, A6 curve, calibration plots, failure gallery, per-tier
   go/no-go).
