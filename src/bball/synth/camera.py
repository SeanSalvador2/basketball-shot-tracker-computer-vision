"""Scene camera for the synthetic engine — a thin layer over lift.projection.Camera.

The pinhole math lives in lift/ (defined and tested once); this adds scene conveniences:
build a camera from a scenario's azimuth/height/distance/FOV, project a whole shot to
image space, and compute apparent ball radius (px) from depth — the quantity the EDA
ball-size analysis reports and the A3 resolution ablation cares about.
"""
from __future__ import annotations

import numpy as np

from bball.lift.projection import Camera
from bball.synth.physics import BALL_DIAMETER_M


def make_camera(
    *,
    width_px: int = 1920,
    height_px: int = 1080,
    hfov_deg: float = 68.0,
    azimuth_deg: float = 45.0,
    height_m: float = 3.0,
    distance_m: float = 9.0,
    hoop_ground_xy=(0.0, 0.0),
) -> Camera:
    """iPhone-wide default (~68 deg HFOV, 1920x1080), aimed at the hoop's ground point."""
    target = (float(hoop_ground_xy[0]), float(hoop_ground_xy[1]), 0.0)
    return Camera.from_look_at(
        width_px=width_px, height_px=height_px, hfov_deg=hfov_deg,
        azimuth_deg=azimuth_deg, height_m=height_m, distance_m=distance_m,
        target=target, target_height_m=1.2,  # aim at ~rim/upper-body height, not the floor
    )


def project_trajectory(camera: Camera, pos3d: np.ndarray) -> np.ndarray:
    """Project an (N,3) court-metre trajectory to (N,2) pixels."""
    return camera.project(np.asarray(pos3d, float))


def apparent_ball_radius_px(camera: Camera, pos3d: np.ndarray) -> np.ndarray:
    """Approximate ball radius in pixels at each 3D position: f * (r_ball / depth)."""
    pts = np.atleast_2d(np.asarray(pos3d, float))
    depth = (camera.R @ (pts - camera.position).T)[2]  # camera-frame Z
    f = camera.K[0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = f * (BALL_DIAMETER_M / 2.0) / depth
    rad[depth <= 1e-6] = np.nan
    return rad


def in_frame(camera: Camera, pts_px: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts_px)
    return (
        (pts[:, 0] >= 0) & (pts[:, 0] < camera.width_px)
        & (pts[:, 1] >= 0) & (pts[:, 1] < camera.height_px)
    )
