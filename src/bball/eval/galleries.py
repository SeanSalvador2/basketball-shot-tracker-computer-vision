"""Failure galleries (plan §8): auto-generated contact sheets of the worst cases per task,
with trajectory overlays. A failure the report can show is a failure understood."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def failure_contact_sheet(cases: list[dict], out_path: str | Path, *, cols: int = 3,
                          title: str = "Failure gallery") -> Path:
    """`cases`: list of {'ball_img': (N,2), 'observed': (N,) bool, 'rim_ellipse': RimEllipse,
    'caption': str}. Writes a PNG contact sheet with each case's image-space trajectory."""
    if not cases:
        # Still write an empty-but-valid figure so the pipeline never breaks.
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no failures", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)

    n = len(cases)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, case in zip(axes, cases):
        img = np.asarray(case["ball_img"], float)
        obs = np.asarray(case.get("observed", ~np.isnan(img).any(axis=1)), bool)
        ax.plot(img[obs, 0], img[obs, 1], ".", color="tab:orange", ms=4)
        br = (~obs) & ~np.isnan(img).any(axis=1)
        if br.any():
            ax.plot(img[br, 0], img[br, 1], "s", mfc="none", color="magenta", ms=4)
        ell = case.get("rim_ellipse")
        if ell is not None:
            poly = ell.boundary_polyline(60)
            ax.plot(poly[:, 0], poly[:, 1], color="k", lw=0.8)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(case.get("caption", ""), fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)
