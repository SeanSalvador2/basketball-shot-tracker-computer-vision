#!/usr/bin/env python
"""Render the three zone-partition presets as court diagrams (reports/figures/zones_presets.png).

Fills are computed by brute-force labeling of a court-plane grid through each partition's
own `label()` — the diagram is generated *from* the shipped logic, so it cannot drift from
the implementation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from bball.lift.court_model import get_court, paint_polygon, three_point_polyline
from bball.lift.zones import preset_basic3, preset_extended, preset_spots

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "zones_presets.png"

FILL = {
    "interior": "#f6c28b",
    "midrange": "#9ecae1", "short-mid": "#c6dbef", "long-mid": "#9ecae1",
    "three": "#a1d99b", "deep-three": "#41ab5d",
}
RANGE_OF_SPOT = lambda z: ("interior" if z == "interior"
                           else "three" if z.endswith("-three") else "midrange")

SPOT_LABELS = {  # annotation anchor per spots-preset zone (right side + center; mirrored)
    "interior": (0.0, 1.0), "top-mid": (0.0, 4.9), "top-three": (0.0, 8.3),
    "right-corner-mid": (4.9, 0.9), "right-wing-mid": (3.7, 3.3),
    "right-corner-three": (7.05, 1.2), "right-wing-three": (5.8, 5.8),
}


def _label_grid(part, court, n=420):
    xs = np.linspace(-court.sideline_x_m, court.sideline_x_m, n)
    ys = np.linspace(-court.rim_from_baseline_m, 10.5, n)
    zone_names = list(dict.fromkeys(part.zones))
    idx = {z: i for i, z in enumerate(zone_names)}
    grid = np.zeros((len(ys), len(xs)), dtype=int)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            grid[r, c] = idx[part.label(float(x), float(y))]
    return xs, ys, grid, zone_names


def _panel(ax, part, court, title, spots_mode=False):
    xs, ys, grid, zone_names = _label_grid(part, court)
    colors = [FILL[RANGE_OF_SPOT(z)] if spots_mode else FILL.get(z, "#dddddd")
              for z in zone_names]
    ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
              cmap=ListedColormap(colors), interpolation="nearest", aspect="equal")
    for poly in part.boundaries.values():
        ax.plot(poly[:, 0], poly[:, 1], color="#333333", lw=1.4)
    tp = three_point_polyline(court)
    ax.plot(tp[:, 0], tp[:, 1], color="#111111", lw=2.0)
    pp = paint_polygon(court)
    ax.plot(pp[:, 0], pp[:, 1], color="#666666", lw=0.9, ls="--")
    ax.plot(0, 0, "o", color="#c1272d", ms=7)  # hoop ground projection
    ax.axhline(-court.rim_from_baseline_m, color="#111111", lw=2.0)
    if spots_mode:
        for z, (x, y) in SPOT_LABELS.items():
            ax.annotate(z, (x, y), ha="center", va="center", fontsize=7.5, weight="bold")
            if z.startswith("right-"):
                ax.annotate(z.replace("right-", "left-"), (-x, y),
                            ha="center", va="center", fontsize=7.5, weight="bold")
    else:
        seen = {}
        for r in range(0, grid.shape[0], 6):
            for c in range(0, grid.shape[1], 6):
                seen.setdefault(zone_names[grid[r, c]], []).append((xs[c], ys[r]))
        for z, pts in seen.items():
            cx, cy = np.mean(pts, axis=0)
            anchor = {"three": (0.0, 8.6), "deep-three": (0.0, 9.9),
                      "interior": (0.0, 1.0), "midrange": (0.0, 4.6),
                      "short-mid": (0.0, 3.2), "long-mid": (0.0, 6.3)}.get(z, (cx, cy))
            ax.annotate(z, anchor, ha="center", va="center", fontsize=9, weight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-court.sideline_x_m - 0.3, court.sideline_x_m + 0.3)
    ax.set_ylim(-court.rim_from_baseline_m - 0.4, 10.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    court = get_court("nba")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    _panel(axes[0], preset_basic3(court),
           court, 'basic3 — interior (≤7 ft) / midrange / three')
    _panel(axes[1], preset_extended(court),
           court, "extended — + short/long-mid split, deep-three (0.9 m offset of the line)")
    _panel(axes[2], preset_spots(court),
           court, "spots — corner / wing / top × mid / three (+ interior)", spots_mode=True)
    fig.suptitle("Zone-partition presets (NBA court spec; hoop at red dot; black = 3PT line)",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
