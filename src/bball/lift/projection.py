"""Pinhole-camera projection primitives (world <-> image), in metric court coordinates.

These are the *forward* geometry that the homography estimator (homography.py) inverts,
and the shared math that synth/camera.py builds scene cameras on. Keeping them here (in
lift/) means the geometry is defined once and unit-tested directly, rather than hidden
inside the renderer.

World frame: court coordinates in metres, right-handed, +Z up, ground plane Z = 0.
Camera frame: OpenCV convention — +X right, +Y down, +Z forward (into the scene).

Error model (plan §5.3): ground-plane position error scales approximately as
    dX  ~=  (h / sin^2(phi)) * (sigma_px / f)
with h the camera height, phi the depression angle to the court point, f the focal
length in pixels, sigma_px the click/localisation noise. Grazing views (small phi) are
hyper-sensitive; elevated views are benign. `predicted_ground_error` implements it for
the A7 sanity check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def intrinsics_from_fov(width_px: int, height_px: int, hfov_deg: float) -> np.ndarray:
    """Camera intrinsic matrix K from horizontal FOV, square pixels, centred principal
    point. Defaults model an iPhone wide lens (~68 deg HFOV at 1920 px)."""
    fx = (width_px / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    cx = width_px / 2.0
    cy = height_px / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def look_at_rotation(eye: np.ndarray, target: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World->camera rotation R (rows are camera axes in world coords) so the camera at
    `eye` looks toward `target`. Camera +Z = forward, +Y = down, +X = right."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    forward = forward / n
    world_up = np.asarray(world_up, float)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        # Looking straight up/down: pick an arbitrary right axis.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    return R


@dataclass(frozen=True)
class Camera:
    """A calibrated pinhole camera positioned in court coordinates."""

    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 world->camera rotation
    position: np.ndarray     # camera centre C in world (metres)
    width_px: int
    height_px: int

    @property
    def t(self) -> np.ndarray:
        """Translation of the [R|t] extrinsic: t = -R C."""
        return -self.R @ self.position

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K [R | t]."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @classmethod
    def from_look_at(
        cls,
        *,
        width_px: int = 1920,
        height_px: int = 1080,
        hfov_deg: float = 68.0,
        azimuth_deg: float = 45.0,
        height_m: float = 3.0,
        distance_m: float = 9.0,
        target=(0.0, 0.0, 0.0),
        target_height_m: float = 0.0,
    ) -> "Camera":
        """Place a camera at a given azimuth/height/distance around a target point.

        `azimuth_deg` is measured in the ground plane about the target: 0 deg looks along
        +Y (baseline behind the hoop, excluded in practice), 90 deg is the sideline.
        `distance_m` is the horizontal (ground) distance from the target.
        """
        target = np.asarray(target, float).copy()
        az = np.radians(azimuth_deg)
        # Camera sits on a circle of radius distance_m around the target, at height_m.
        eye = target + np.array([distance_m * np.sin(az), -distance_m * np.cos(az), height_m])
        look = target.copy()
        look[2] = target_height_m
        K = intrinsics_from_fov(width_px, height_px, hfov_deg)
        R = look_at_rotation(eye, look)
        return cls(K=K, R=R, position=eye, width_px=width_px, height_px=height_px)

    # -- projection --------------------------------------------------------
    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixel coordinates (may fall outside the frame;
        points behind the camera return NaN)."""
        pts = np.atleast_2d(np.asarray(points_world, float))
        cam = (self.R @ (pts - self.position).T).T  # Nx3 in camera frame
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * cam[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * cam[:, 1] / z + self.K[1, 2]
        out = np.stack([u, v], axis=1)
        out[z <= 1e-9] = np.nan  # behind or on the camera plane
        return out

    def ground_homography(self) -> np.ndarray:
        """3x3 homography mapping ground-plane points [X, Y, 1] to image [u, v, 1].

        Derivation: for Z = 0, K(R[:,0] X + R[:,1] Y + t) = K [R[:,0] R[:,1] t] [X Y 1]^T.
        """
        cols = np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        H = self.K @ cols
        return H / H[2, 2]

    def depression_angle(self, ground_point) -> float:
        """Depression angle phi (radians) of the ray to a ground point: sin(phi) = h/dist."""
        p = np.asarray(ground_point, float)
        if p.shape[0] == 2:
            p = np.array([p[0], p[1], 0.0])
        ray = p - self.position
        dist = np.linalg.norm(ray)
        drop = self.position[2] - p[2]
        return float(np.arcsin(np.clip(drop / max(dist, 1e-9), -1.0, 1.0)))

    def predicted_ground_error(self, ground_point, sigma_px: float) -> float:
        """Analytic ground-error estimate from the h/sin^2(phi) model (metres)."""
        phi = self.depression_angle(ground_point)
        f = self.K[0, 0]
        h = self.position[2]
        s = np.sin(phi)
        if s < 1e-6:
            return float("inf")
        return float(h / (s * s) * (sigma_px / f))


def project_points(camera: Camera, points_world: np.ndarray) -> np.ndarray:
    return camera.project(points_world)
