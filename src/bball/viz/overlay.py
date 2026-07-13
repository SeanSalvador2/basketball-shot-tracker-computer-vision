"""Overlay rendering: detections/track/trajectory/rim on frames, and image trajectories
mapped onto the court."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def overlay_frame(frame_bgr: np.ndarray, *, ball_xy=None, rim_ellipse=None, boxes=None,
                  bridged_xy=None, ax=None, title=None):
    """Draw an annotated frame (expects BGR; converts to RGB for display)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(frame_bgr[:, :, ::-1])
    if rim_ellipse is not None:
        poly = rim_ellipse.boundary_polyline(80)
        ax.plot(poly[:, 0], poly[:, 1], color="yellow", lw=1.5)
    if boxes:
        for (x0, y0, x1, y1) in boxes:
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", lw=1)
    if ball_xy is not None and not np.isnan(ball_xy).any():
        ax.plot(ball_xy[0], ball_xy[1], "o", color="tab:orange", ms=8)
    if bridged_xy is not None and not np.isnan(bridged_xy).any():
        ax.plot(bridged_xy[0], bridged_xy[1], "s", color="magenta", ms=6, mfc="none")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_trajectory_on_court(traj3d_positions: np.ndarray, court="nba", ax=None, *,
                             label=None, color="tab:blue"):
    """Plot a reconstructed 3D trajectory's ground track on the half court."""
    from bball.viz.court import plot_halfcourt

    ax = plot_halfcourt(court, ax=ax)
    pos = np.asarray(traj3d_positions, float)
    ax.plot(pos[:, 0], pos[:, 1], "-", color=color, alpha=0.7, label=label)
    if label:
        ax.legend(fontsize=8)
    return ax


def plot_image_trajectory(ball_img: np.ndarray, observed: np.ndarray | None = None,
                          rim_ellipse=None, ax=None, title=None):
    """Plot an image-space ball trajectory, distinguishing observed vs bridged points."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    img = np.asarray(ball_img, float)
    if observed is None:
        observed = ~np.isnan(img).any(axis=1)
    observed = np.asarray(observed, bool)
    ax.plot(img[observed, 0], img[observed, 1], ".", color="tab:orange", label="observed")
    br = (~observed) & ~np.isnan(img).any(axis=1)
    if br.any():
        ax.plot(img[br, 0], img[br, 1], "s", mfc="none", color="magenta", label="bridged")
    if rim_ellipse is not None:
        poly = rim_ellipse.boundary_polyline(80)
        ax.plot(poly[:, 0], poly[:, 1], color="k", lw=1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax
