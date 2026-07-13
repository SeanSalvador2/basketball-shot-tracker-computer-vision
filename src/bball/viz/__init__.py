"""VIZ — court plots, overlays, reliability diagrams, timelines.

Figures render headless on the non-interactive Agg backend so every report figure is
regenerable from a committed config + seed (plan gate G5). Inside a Jupyter kernel the
inline backend is left alone so notebook cells still capture their figures.
"""
import sys

import matplotlib

if "ipykernel" not in sys.modules:  # scripts/tests: headless Agg; notebooks: keep inline
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
