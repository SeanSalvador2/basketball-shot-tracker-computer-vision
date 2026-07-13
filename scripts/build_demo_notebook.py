#!/usr/bin/env python3
"""Generate notebooks/demo.ipynb from cell definitions (kept in code so the notebook is
reproducible and reviewable). Run once; `make demo` then executes the notebook end-to-end.

    python scripts/build_demo_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s))

    md("""# Basketball Shot Tracker — end-to-end demo

**Regime: S (synthetic).** This notebook runs the full **DETECT → TRACK → LIFT → CLASSIFY**
pipeline on a bundled synthetic clip and visualizes every stage: ball detections (classical
background subtraction — *zero downloaded weights*), the bridged ball trajectory, the
rim-normalized make/miss FSM, the court-mapped shot chart, and calibrated make probabilities.

The pipeline is **real-footage-ready**: the only session-specific inputs are the clip and the
one-time calibration (rim ellipse + homography). On real footage those come from the session
setup; here they are the synthetic ground truth. No number here is a real-world accuracy
claim — Stage B fills those in.""")

    code("""import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from bball.demo import load_demo, run_demo, accuracy_vs_gt
from bball.viz.overlay import overlay_frame, plot_image_trajectory
from bball.viz.court import shot_chart

ASSETS = Path("assets")
frames, meta, rim, H_img_to_court, court = load_demo(ASSETS)
print(f"loaded {len(frames)} frames @ {meta['fps']} fps, {len(meta['shots'])} shots, "
      f"render {frames[0].shape[1]}x{frames[0].shape[0]}")""")

    md("### 1. A frame with the annotated rim ellipse\nThe rim ellipse is the projective image of the rim circle; the FSM's predicates are expressed relative to it (rim-normalized).")
    code("""sm = meta['shots'][0]
mid = (sm['frame_start'] + sm['frame_end']) // 2
fig, ax = plt.subplots(figsize=(8, 4.5))
overlay_frame(frames[mid], rim_ellipse=rim, ax=ax, title="rendered frame + rim ellipse")
plt.show()""")

    md("### 2. Detect → track → bridge (one shot)\nBackground subtraction proposes ball candidates; the two-level trajectory layer bridges the occlusion gap where the ball vanishes into the rim/net. Orange = observed detections, magenta squares = bridged (predicted) points.")
    code("""result = run_demo(ASSETS)
ps = result['per_shot'][0]
br = ps['art']['bridged']
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_image_trajectory(br.xy, br.observed, rim_ellipse=rim, ax=ax,
                      title=f"shot 1 ball track — completeness {br.completeness:.0%}, "
                            f"{len(br.gaps)} bridged run(s)")
plt.show()""")

    md("### 3. Make/miss FSM + calibrated probability, per shot\nThe FSM emits a terminal-state verdict and a margin score; the margin maps to a make probability.")
    code("""print(f"{'shot':>4} {'zone':>12} {'GT':>5} {'pred':>5} {'P(make)':>8}")
for i, ps in enumerate(result['per_shot']):
    r, gt = ps['result'], ps['meta']['gt_outcome']
    flag = '' if r.outcome == gt else '  <-- mismatch'
    print(f"{i+1:>4} {r.zone:>12} {gt:>5} {r.outcome:>5} {r.make_prob:>8.2f}{flag}")
acc = accuracy_vs_gt(result)
print(f"\\ndemo make/miss accuracy vs ground truth: {acc['correct']}/{acc['n']} = {acc['accuracy']:.0%} (regime S)")""")

    md("### 4. Court-mapped shot chart\nShooter positions are lifted to court coordinates (here the synthetic GT feet; Stage B tracks the shooter). Green ● = make, red ✕ = miss.")
    code("""fig, ax = plt.subplots(figsize=(6, 6))
shot_chart(result['report'].shot_chart_data(), court=court, ax=ax,
           title=f"Session shot chart — FG {result['report'].fg_pct():.0%} "
                 f"({result['report'].n_makes}/{result['report'].n_attempts})")
plt.show()
import json
print(json.dumps(result['report'].summary(), indent=2))""")

    md("""### Notes on honesty & scope
- **Detection** here is background subtraction (no weights) so the demo is fully
  self-contained; swapping in a fine-tuned detector is a one-line change (same `BallCandidate`
  contract).
- **Short/long** miss direction and near-rim make/miss are camera-placement dependent (see
  ablation **A6**); this clip uses a 1.5 m tripod at ~55° where the rim ellipse is well-formed
  (see EDA `eda_rim_geometry`).
- Every number is **synthetic (S)**. The same commands run on real footage in Stage B; no
  real permissively-licensed fixed-camera half-court clip with a known calibration was
  available in-container, so the demo is synthetic and the pipeline is real-footage-ready.""")

    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    out = REPO / "notebooks" / "demo.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"[demo] wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
