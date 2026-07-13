"""Court plots and shot charts (hoop-centred metres)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.lift.court_model import CourtSpec, get_court, paint_polygon, three_point_polyline


def plot_halfcourt(court: CourtSpec | str = "nba", ax=None, *, color="0.4"):
    court = get_court(court) if isinstance(court, str) else court
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    sx = court.sideline_x_m
    yb = -court.rim_from_baseline_m
    ytop = court.three_arc_radius_m + 2.0
    ax.plot([-sx, sx], [yb, yb], color=color)                       # baseline
    ax.plot([-sx, -sx], [yb, ytop], color=color)
    ax.plot([sx, sx], [yb, ytop], color=color)
    paint = paint_polygon(court)
    ax.plot(paint[:, 0], paint[:, 1], color=color)
    three = three_point_polyline(court)
    ax.plot(three[:, 0], three[:, 1], color=color)
    ax.add_patch(plt.Circle((0, 0), court.restricted_radius_m, fill=False, color=color, lw=0.8))
    ax.plot(0, 0, marker="o", color="tab:red", ms=6)               # hoop
    ax.set_aspect("equal")
    ax.set_xlim(-sx - 0.5, sx + 0.5)
    ax.set_ylim(yb - 0.5, ytop + 0.5)
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    return ax


def shot_chart(shots: list[dict], court: CourtSpec | str = "nba", ax=None, *, title=None):
    """`shots`: list of {'xy': (x,y), 'outcome': 'make'|'miss'}. Makes = filled, misses = x."""
    ax = plot_halfcourt(court, ax=ax)
    for s in shots:
        x, y = s["xy"]
        if s.get("outcome") == "make":
            ax.plot(x, y, "o", color="tab:green", ms=7, mfc="tab:green", alpha=0.8)
        else:
            ax.plot(x, y, "x", color="tab:red", ms=8, mew=2, alpha=0.8)
    if title:
        ax.set_title(title)
    return ax


def zone_accuracy_bar(zone_acc: dict, ax=None):
    """Bar chart of per-zone shooting percentage from a {zone: (makes, attempts)} dict."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    zones = list(zone_acc.keys())
    pct = [100 * m / max(a, 1) for (m, a) in zone_acc.values()]
    ax.bar(zones, pct, color="tab:blue")
    ax.set_ylabel("FG %")
    ax.set_ylim(0, 100)
    return ax
