"""VIZ — court plots, overlays, reliability diagrams, timelines.

All figures use the non-interactive Agg backend so they render headless and are
regenerable from a committed config + seed (plan gate G5).
"""
import matplotlib

matplotlib.use("Agg")

from bball.viz.court import plot_halfcourt, shot_chart  # noqa: E402
from bball.viz.reliability import reliability_diagram  # noqa: E402
from bball.viz.overlay import overlay_frame, plot_trajectory_on_court  # noqa: E402

__all__ = [
    "plot_halfcourt",
    "shot_chart",
    "reliability_diagram",
    "overlay_frame",
    "plot_trajectory_on_court",
]
