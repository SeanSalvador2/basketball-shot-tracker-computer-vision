"""Reliability diagrams (plan §8, ablation A9)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.events.calibration import expected_calibration_error, reliability_curve


def reliability_diagram(prob_sets: dict, labels: np.ndarray, *, n_bins: int = 10, ax=None, title=None):
    """`prob_sets`: {name: probs} (e.g. {'uncalibrated':..., 'temperature':...}). Plots each
    curve against the diagonal with its ECE in the legend."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, probs in prob_sets.items():
        centers, acc, conf, count = reliability_curve(probs, labels, n_bins)
        ok = count > 0
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.plot(conf[ok], acc[ok], "o-", label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title)
    return ax
