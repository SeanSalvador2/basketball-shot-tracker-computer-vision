#!/usr/bin/env python
"""Labeled map of the calibration landmarks (reports/figures/calibration_landmarks.png).

Generated from `court_model.landmark_points` so the map cannot drift from what the
workbench's dropdown offers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bball.lift.court_model import get_court, landmark_points, paint_polygon, three_point_polyline

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "calibration_landmarks.png"

# label offsets (metres) so the crowded baseline row stays readable
OFF = {
    "baseline_left_corner": (-1.2, -0.75),
    "baseline_right_corner": (-1.0, -0.75), "lane_baseline_left": (-1.4, -1.5),
    "lane_baseline_right": (-1.2, -1.5), "corner_three_left": (-0.6, 0.55),
    "corner_three_right": (-1.6, 0.55), "ft_left": (-3.3, 0.0), "ft_right": (0.4, 0.0),
    "ft_center": (-1.2, -0.65), "top_of_key": (0.45, 0.1), "three_apex": (0.45, 0.1),
}


def main() -> None:
    c = get_court("nba")
    lms = landmark_points(c)
    fig, ax = plt.subplots(figsize=(9.5, 10))
    tp = three_point_polyline(c)
    ax.plot(tp[:, 0], tp[:, 1], "k-", lw=2)
    pp = paint_polygon(c)
    ax.plot(pp[:, 0], pp[:, 1], ls="-", lw=1.4, color="#555555")
    yb, sx = -c.rim_from_baseline_m, c.sideline_x_m
    ax.plot([-sx, sx], [yb, yb], "k-", lw=2.4)
    for s in (-sx, sx):
        ax.plot([s, s], [yb, 11.0], "k-", lw=1.4)
    # FT circle (upper half) for the top_of_key context
    import numpy as np

    t = np.linspace(0, np.pi, 60)
    ax.plot(1.8288 * np.cos(t), c.ft_line_from_hoop_m + 1.8288 * np.sin(t),
            ls="--", lw=1.2, color="#777777")
    ax.plot([-0.9, 0.9], [-0.35, -0.35], "-", color="#333333", lw=3)  # backboard

    for i, (name, xy) in enumerate(lms.items(), 1):
        ax.plot(xy[0], xy[1], "o", ms=11, color="#c1272d", mec="black", zorder=5)
        ax.annotate(str(i), xy, ha="center", va="center", color="white",
                    fontsize=7.5, weight="bold", zorder=6)
        dx, dy = OFF.get(name, (0.4, 0.1))
        ax.annotate(f"{i}. {name}", (xy[0] + dx, xy[1] + dy), fontsize=9.5,
                    weight="bold", color="#1a3a5c", va="center")
    ax.set_title("Calibration landmarks — click each crisply visible one in the workbench\n"
                 "(numbers match the dropdown order; skip any you can't see sharply)",
                 fontsize=11)
    ax.set_xlim(-sx - 1.6, sx + 3.2)
    ax.set_ylim(yb - 1.6, 11.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
