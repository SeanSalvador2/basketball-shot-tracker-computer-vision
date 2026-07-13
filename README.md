# Basketball Shot Tracker — Computer Vision

A phone films a half basketball court from a fixed wide angle; this system counts makes and misses, maps every shot to its court location, and — in progressively ambitious tiers — classifies shot type, miss direction, and make quality.

The organizing spine:

**DETECT** (ball, rim, player) → **TRACK** (temporal association + ballistic occlusion bridging) → **LIFT** (one-time homography to real court coordinates) → **CLASSIFY** (event state machine + small learned heads).

Two structural facts drive the design: the **camera is fixed**, so court registration is a one-time 4–8 point homography rather than a per-frame learned-calibration problem; and capture is **record-then-process**, so inference cost is a budget, not a wall.

## Project status

| Phase | Status | Artifact |
|---|---|---|
| Phase 0 — Literature & SOTA survey | ✅ complete | [`reports/phase0_research.md`](reports/phase0_research.md) |
| Project plan (full DS lifecycle, ablation matrix, staged execution) | ✅ complete (v1.1) | [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) |
| Adversarial plan review (12 findings → revisions) | ✅ complete | [`docs/PLAN_REVIEW.md`](docs/PLAN_REVIEW.md) |
| Phase 1 — Research pipeline, experiments, ablations | 🚧 in progress | `reports/`, `src/bball/` |
| Phase 2 — On-device deployment (Core ML, iPhone) | outlined | plan §9.2 |

## Why this is not just a YOLO wrapper

- **License-driven architecture:** the entire Ultralytics YOLO ecosystem is excluded (AGPL-3.0 extends to self-trained weights); every shipped dependency is Apache/MIT/BSD, tracked in a license ledger.
- **Geometry does the heavy lifting:** near-rim event logic is *rim-normalized* (predicates relative to the projective image of the rim circle), shot location rides a calibrated homography with a derived error model — no raw-pixel thresholds anywhere.
- **Honest observability:** a single camera cannot observe an airborne ball's 3D position; the design separates image-space trajectory fitting (association, bridging) from confidence-gated 3D ballistic reconstruction (miss direction), and reports per-axis accuracy instead of hiding the depth collapse.
- **Evaluation discipline:** session-level splits (never random frames), a held-out cross-venue test set, calibrated probabilities (reliability diagrams, ECE, Brier), stratified error analysis, and a stated ablation matrix with hypotheses.

## Repository map

```
docs/       project plan, plan review, data protocol
reports/    phase reports + figures (Phase 0 research survey lives here)
src/bball/  pipeline package: detect / track / lift / events / heads / synth / eval / viz
configs/    experiment configs (every experiment is a committed config)
notebooks/  end-to-end demo notebook
tests/      geometry, event-logic, and anti-leakage tests
```

*Full details, targets, and the ablation matrix: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).*
