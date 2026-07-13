#!/usr/bin/env python
"""Camera-placement guide (reports/figures/camera_placement_guide.png).

Top-down panel: where to put the tripod, annotated with the measured A6 per-axis
miss-direction tradeoff (azimuth is the angle off the shooting lane; 90° = sideline).
Side panel: heights — the A7 error model rewards elevation (ground error ~ 1/sin^2(phi)),
but the EDA rim-geometry finding forbids the ~rim-height band (rim images edge-on).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bball.lift.court_model import get_court, paint_polygon, three_point_polyline

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "camera_placement_guide.png"
R_CAM = 12.0  # tripod distance from hoop in the sketch (m)


def _court(ax):
    c = get_court("nba")
    yb, sx = -c.rim_from_baseline_m, c.sideline_x_m
    ax.plot([-sx, sx], [yb, yb], "k-", lw=2)                      # baseline
    for s in (-sx, sx):
        ax.plot([s, s], [yb, 13.0], "k-", lw=1.2)                 # sidelines
    tp = three_point_polyline(c)
    ax.plot(tp[:, 0], tp[:, 1], "k-", lw=1.5)
    pp = paint_polygon(c)
    ax.plot(pp[:, 0], pp[:, 1], ls="--", lw=0.8, color="#777777")
    ax.plot(0, 0, "o", color="#c1272d", ms=8)
    ax.plot([-0.9, 0.9], [-0.35, -0.35], "-", color="#444444", lw=2.5)  # backboard
    return c


def _cam_xy(az_deg: float) -> tuple[float, float]:
    """Camera position at R_CAM from the hoop; azimuth measured off the shooting lane on
    the behind-the-hoop side (0 = straight behind the backboard, 90 = sideline)."""
    a = np.deg2rad(az_deg)
    return R_CAM * np.sin(a), -R_CAM * np.cos(a)


def top_down(ax):
    _court(ax)
    spots = [
        (15, "#e08214", "^", "15°\nL/R best (0.96)\nS/L weakest"),
        (30, "#4393c3", "o", "30°\nL/R 0.93 · S/L 0.74"),
        (45, "#1a7837", "*", "45°  DEFAULT"),
        (60, "#1a7837", "*", "60°  DEFAULT\n(45–60° balances both axes)"),
        (90, "#4393c3", "o", "90° sideline\nS/L perfect (1.00)\nL/R weakest (0.75)"),
    ]
    for az, color, marker, label in spots:
        x, y = _cam_xy(az)
        ax.plot(x, y, marker, color=color, ms=16 if marker == "*" else 11,
                mec="black", mew=0.8, zorder=5)
        ax.plot([0, x], [0, y], ":", color=color, lw=0.9, alpha=0.7)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, -2),
                    fontsize=8, weight="bold", color=color, va="center")
    xb, yb_ = _cam_xy(0)
    ax.plot(xb, yb_, "X", color="#b2182b", ms=13, mec="black", zorder=5)
    ax.annotate("0° straight behind:\nDON'T (backboard occludes\nthe rim approach)",
                (xb, yb_), textcoords="offset points", xytext=(-8, -6),
                fontsize=8, weight="bold", color="#b2182b", ha="right", va="top")
    ax.annotate("mirroring to the other side is equivalent —\npick whichever side has room",
                (-7.2, -10.5), fontsize=8, style="italic", color="#555555")
    ax.set_title("WHERE (top-down) — azimuth off the shooting lane; ~10–13 m from the hoop\n"
                 "Tier-1 minimum: one session each at ≈30°, 45–60°, 90°", fontsize=10)
    ax.set_xlim(-12.5, 18.5)
    ax.set_ylim(-14.5, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")


def side_view(ax):
    ax.axhline(0, color="black", lw=2)                                   # floor
    ax.plot([0, 0], [0, 3.35], "-", color="#555555", lw=3)               # pole
    ax.plot([-0.05, 0.6], [3.05, 3.05], "-", color="#c1272d", lw=3)      # rim @ 3.05
    ax.plot([-0.05, -0.05], [2.9, 3.95], "-", color="#333333", lw=2.5)   # backboard
    ax.annotate("rim 3.05 m", (0.7, 3.05), fontsize=8, va="center")
    ax.axhspan(2.7, 3.3, color="#b2182b", alpha=0.15)
    ax.annotate("NEVER 2.7–3.3 m: camera at rim height sees the rim edge-on\n"
                "(rim ellipse collapses — EDA finding)", (1.2, 2.3),
                fontsize=8.5, weight="bold", color="#b2182b", va="top")
    cams = [
        (1.5, "#1a7837", "1.5 m tripod — the workhorse ✓"),
        (4.5, "#1a7837", "≥ 4 m (balcony/fence) — location error shrinks ~1/sin²φ ✓✓\n"
                          "one session here if you can"),
    ]
    for h, color, label in cams:
        ax.plot(12.0, h, "s", color=color, ms=11, mec="black", zorder=5)
        ax.plot([12.0, 0.3], [h, 3.05], ":", color=color, lw=1.0)
        ax.plot([12.0, 4.0], [h, 0.0], ":", color=color, lw=1.0, alpha=0.6)
        ax.annotate(label, (12.4, h), fontsize=8.5, weight="bold", color=color, va="center")
    ax.plot(12.0, 3.0, "X", color="#b2182b", ms=12, mec="black", zorder=5)
    ax.set_title("HOW HIGH (side view at ~12 m)", fontsize=10)
    ax.set_xlim(-1.5, 22)
    ax.set_ylim(-0.6, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    top_down(ax1)
    side_view(ax2)
    fig.suptitle("Camera placement guide — grounded in the A6 azimuth sweep + A7 error model",
                 fontsize=12, y=0.995)
    fig.text(0.5, 0.015,
             "Every session: LANDSCAPE orientation (locked) · 1× main lens (not ultrawide) · "
             "1080p60 · lock AE/AF (long-press)\nFrame: whole half court + rim in the upper "
             "third + clear air above the rim (the arc apex flies 1–2 m above it)",
             ha="center", fontsize=9.5, weight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
