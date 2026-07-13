# Basketball Shot Tracker — Computer Vision

A phone films a half basketball court from a fixed wide angle; this system counts makes and
misses, maps every shot to its court location, and — in progressively ambitious tiers —
classifies shot type, miss direction, and make quality.

The organizing spine:

**DETECT** (ball, rim, player) → **TRACK** (temporal association + ballistic occlusion
bridging) → **LIFT** (one-time homography to real court coordinates) → **CLASSIFY** (event
state machine + small learned heads).

Two structural facts drive the design: the **camera is fixed**, so court registration is a
one-time 4–8 point homography rather than a per-frame learned-calibration problem; and
capture is **record-then-process**, so inference cost is a budget, not a wall.

## Project status

| Phase | Status | Artifact |
|---|---|---|
| Phase 0 — Literature & SOTA survey | ✅ complete | [`reports/phase0_research.md`](reports/phase0_research.md) |
| Project plan (full DS lifecycle, ablation matrix, staged execution) | ✅ complete (v1.1) | [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) |
| Adversarial plan review (12 findings → revisions) | ✅ complete | [`docs/PLAN_REVIEW.md`](docs/PLAN_REVIEW.md) |
| **Phase 1 · Stage A** — pipeline, synthetic engine, ablations (CPU container) | ✅ **complete** | [`reports/phase1_summary.md`](reports/phase1_summary.md) |
| Phase 1 consolidated final report (start here after this README) | ✅ complete | [`reports/phase1_final_report.md`](reports/phase1_final_report.md) |
| Configurable shot-zone system (presets / parametric / screen-drawn) | ✅ complete | [`docs/ZONES.md`](docs/ZONES.md) |
| **Web workbench app** — calibrate / review-label / zone editor / shot chart (PWA, phone-usable) | ✅ complete (`make app`) | [`docs/APP.md`](docs/APP.md) |
| Native iOS app (on-device Core ML) | Phase 2 (post-checkpoint; needs macOS/Xcode) | plan §9.2 |
| Phase 1 · Stage B — real footage, detector fine-tuning, cross-venue numbers | next (runbook ready) | [`docs/DATA_PROTOCOL.md`](docs/DATA_PROTOCOL.md) |
| Phase 2 — On-device deployment (Core ML, iPhone) | outlined | plan §9.2 |

## Results (Stage A — regime **S = synthetic**; R = real arrives with Stage B)

Synthetic results validate *logic and geometry*, never real-world perception accuracy —
that discipline is the point. Full analysis:
[`reports/phase1_experiments.md`](reports/phase1_experiments.md).

| Result | Value | Regime | Where |
|---|---|---|---|
| Shot-location error, ≥6-point homography at careful (2 px) clicks | **4.7 cm median / 11 cm P90**, zone acc ≥0.99 | S | A7 |
| Same at sloppy (5 px) clicks, 4 → 8 points | 34.7 → **9.8 cm** median | S | A7 |
| Make/miss F1 across the 36-cell FSM parameter grid | **0.86–0.98** (a plateau, not a knife-edge) | S | A8 |
| FSM batch accuracy on ground-truth tracks, 4 camera placements | **89–96%** | S | experiments report |
| Occlusion bridging (Level-1) vs 3→30-frame rim gaps | F1 **0.96→0.72**; no-bridging erratic; Level-2 fill *worse* (kept for metric outputs only) | S | A5 |
| Miss-direction accuracy vs camera placement (headline curve) | axes **trade** with azimuth — L/R 0.96@15°→0.75@90°, S/L 0.74@30°→1.00@90°; **45–60° balances both** | S | A6 |
| Probability calibration on a held-out venue | Platt cuts ECE **0.117→0.027** (−77%); temperature scaling fails on FSM margins (negative result, kept) | S | A9 |
| End-to-end demo, zero downloaded weights | **5/6** shots correct; the miss is a narrated point-blank visibility case | S | `make demo` |
| COCO-pretrained detector on synthetic renders vs real images | **0.0 ball recall on renders at any resolution** (stylized synthetic cannot evaluate a real detector — the regime-label discipline, proven); zero-shot on real images: ball fire-rate 0.30–0.35, person 0.95 | S + R-zeroshot | A3 |
| Player-tracker sophistication in the waypoint sim | greedy-IoU **0.715** HOTA vs Kalman+IoU 0.578 (hard direction changes break constant-velocity); simplified HOTA, real footage arbitrates | S | A4 |

Figure highlights: `reports/figures/ablations/a6_azimuth_sweep.png` (the camera-placement
guide), `a7_error_isolines.png` (court error map), `a8_fsm_sensitivity.png`,
`a9_reliability.png`, and the EDA set under `reports/figures/eda/` — including the
build-surfaced finding that a camera mounted at ≈ rim height images the rim edge-on and
should be avoided (`eda_rim_geometry.png`).

## Quickstart

```bash
make setup      # venv + pinned deps (torch CPU wheels) + editable install
make test       # 128 tests: geometry, FSM scenarios, bridging, zones, app API, leakage guards
make app        # local web workbench: calibrate / label / draw zones / shot chart (docs/APP.md)
make demo       # execute notebooks/demo.ipynb end-to-end on the bundled synthetic clip
make eda        # regenerate every EDA figure
make ablations  # re-run the ablation suite (MLflow file store + committed exports)
```

Every figure/number maps to one command: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Why this is not just a YOLO wrapper

- **License-driven architecture:** the entire Ultralytics YOLO ecosystem is excluded
  (AGPL-3.0 extends to self-trained weights); every shipped dependency is Apache/MIT/BSD,
  tracked in [`LICENSES.md`](LICENSES.md).
- **Geometry does the heavy lifting:** near-rim event logic is *rim-normalized* (predicates
  relative to the projective image of the rim circle), shot location rides a calibrated
  homography with a derived error model — no raw-pixel thresholds anywhere.
- **Honest observability:** a single camera cannot observe an airborne ball's 3D position;
  the design separates image-space trajectory fitting (association, bridging) from
  confidence-gated 3D reconstruction (miss direction), and reports per-axis accuracy
  instead of hiding the depth collapse — the A6 curve quantifies it.
- **Evaluation discipline:** session-level splits (never random frames), a held-out
  cross-venue test set, val-tune/val-cal separation, calibrated probabilities (reliability
  diagrams, ECE, Brier), per-session bootstrap CIs, and registered hypotheses — including
  the ones the data rejected (A9's temperature scaling; A5's Level-2 bridging; A6's
  flat-left/right assumption).
- **Zones are a view, not a measurement:** shot positions are stored continuously; user
  taxonomies (presets, parametric bounds, screen-drawn regions lifted to court space) re-bucket
  history instantly, with per-boundary reliability scored against the calibration error model
  ([`docs/ZONES.md`](docs/ZONES.md)).

## Repository map

```
docs/           project plan, plan review, data protocol, reproducing guide
reports/        phase reports + figures (EDA, ablations)
src/bball/      pipeline package: detect / track / lift / events / heads / synth /
                eval / viz / ablations
configs/        experiment configs (every experiment is a committed config)
scripts/        ablation runner, demo builders, event-review CLI, Stage-B data downloaders
notebooks/      end-to-end demo notebook (`make demo`) + bundled synthetic clip
tests/          120 tests: geometry, events, tracking, synth, zones, eval, leakage guards
mlruns-export/  committed CSV/JSON summaries of tracked runs (MLflow IDs inside)
```

*Full details, targets, and the ablation matrix: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).*
