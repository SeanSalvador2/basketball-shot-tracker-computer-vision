"""Rim frame: the annotated rim ellipse and the rim 3D anchor.

The rim is a horizontal circle of known radius at known height (3.048 m). Under a pinhole
camera its image is an **ellipse** — and that ellipse is the projective image of the rim
circle, so expressing near-rim positions as fractions of the ellipse axes absorbs the
camera geometry (review R2). This module holds:

* `RimEllipse` — the annotation (centre, semi-axes, rotation) + rim-normalized coordinate
  transform used by the FSM (bball.events.fsm).
* `fit_ellipse` — a numerically stable direct ellipse fit (Halir-Flusser) used to derive
  a rim ellipse from projected circle points (synthetic annotation) and cross-check
  cv2.fitEllipse in tests.
* `rim_3d_center` / `rim_circle_3d` — the metric rim anchor for Level-2 reconstruction
  (bball.track.ballistic), from the rim's ground projection + known height.
* `RimAnnotation` — the per-session on-disk format (JSON) and its loader.

Rim is not a COCO class; per-session manual ROI/ellipse annotation is a deliberate,
documented non-problem for the fixed camera (plan §5.1).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Ellipse fitting (Halir & Flusser 1998, numerically stable direct least squares)
# --------------------------------------------------------------------------- #
def fit_ellipse(points: np.ndarray) -> np.ndarray:
    """Fit a general conic ellipse to Nx2 points; returns conic coeffs [A,B,C,D,E,F]
    for A x^2 + B x y + C y^2 + D x + E y + F = 0."""
    pts = np.asarray(points, float)
    if pts.shape[0] < 5:
        raise ValueError("ellipse fit needs >= 5 points")
    x, y = pts[:, 0], pts[:, 1]
    D1 = np.stack([x * x, x * y, y * y], axis=1)
    D2 = np.stack([x, y, np.ones_like(x)], axis=1)
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    C1inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])
    M = C1inv @ M
    eigvals, eigvecs = np.linalg.eig(M)
    cond = 4 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
    a1 = None
    for k in range(3):
        if np.isreal(cond[k]) and cond[k].real > 0:
            a1 = np.real(eigvecs[:, k])
            break
    if a1 is None:
        raise ValueError("ellipse fit failed (no valid conic; points may be collinear)")
    a2 = T @ a1
    return np.concatenate([a1, a2])


def conic_to_geometric(conic: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Convert conic coeffs to (center[2], semi_major, semi_minor, theta_rad of major axis)."""
    A, B, C, D, E, F = conic
    A2 = np.array([[A, B / 2.0], [B / 2.0, C]])
    if np.linalg.det(A2) <= 0:
        raise ValueError("conic is not an ellipse (det of quadratic part <= 0)")
    center = np.linalg.solve(A2, [-D / 2.0, -E / 2.0])
    cx, cy = center
    Fp = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F
    eigvals, eigvecs = np.linalg.eigh(A2)
    axes = np.sqrt(np.maximum(-Fp / eigvals, 0.0))
    if axes[0] >= axes[1]:
        semi_major, semi_minor = axes[0], axes[1]
        major_vec = eigvecs[:, 0]
    else:
        semi_major, semi_minor = axes[1], axes[0]
        major_vec = eigvecs[:, 1]
    theta = float(np.arctan2(major_vec[1], major_vec[0]))
    return center, float(semi_major), float(semi_minor), theta


@dataclass(frozen=True)
class RimEllipse:
    """Rim ellipse in image (pixel) coordinates. `a` = semi-major, `b` = semi-minor,
    `theta_deg` = major-axis orientation."""

    cx: float
    cy: float
    a: float
    b: float
    theta_deg: float

    @classmethod
    def from_points(cls, points: np.ndarray) -> "RimEllipse":
        center, a, b, theta = conic_to_geometric(fit_ellipse(points))
        return cls(cx=float(center[0]), cy=float(center[1]), a=a, b=b, theta_deg=float(np.degrees(theta)))

    def to_normalized(self, pts: np.ndarray) -> np.ndarray:
        """Map image points into the rim-normalized frame: translate to the ellipse
        centre, rotate into its axes, divide by the semi-axes. Inside the ellipse iff
        nx^2 + ny^2 < 1."""
        pts = np.atleast_2d(np.asarray(pts, float))
        th = np.radians(self.theta_deg)
        c, s = np.cos(-th), np.sin(-th)
        d = pts - np.array([self.cx, self.cy])
        dx = c * d[:, 0] - s * d[:, 1]
        dy = s * d[:, 0] + c * d[:, 1]
        return np.stack([dx / self.a, dy / self.b], axis=1)

    def radial_fraction(self, pts: np.ndarray) -> np.ndarray:
        """sqrt(nx^2 + ny^2): 0 at centre, 1 on the boundary, >1 outside."""
        n = self.to_normalized(pts)
        return np.sqrt((n**2).sum(axis=1))

    def contains(self, pts: np.ndarray) -> np.ndarray:
        return self.radial_fraction(pts) < 1.0

    def interior_margin(self, pts: np.ndarray) -> np.ndarray:
        """1 - radial_fraction: positive inside, 0 on the boundary, negative outside — the
        raw material for the FSM margin score."""
        return 1.0 - self.radial_fraction(pts)

    def boundary_polyline(self, n: int = 100) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n)
        th = np.radians(self.theta_deg)
        c, s = np.cos(th), np.sin(th)
        ex = self.a * np.cos(t)
        ey = self.b * np.sin(t)
        x = self.cx + c * ex - s * ey
        y = self.cy + s * ex + c * ey
        return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------- #
# Rim 3D anchor (metric)
# --------------------------------------------------------------------------- #
def rim_3d_center(rim_ground_xy, rim_height_m: float = 3.048) -> np.ndarray:
    """3D rim centre in court coords from its ground projection + known height."""
    g = np.asarray(rim_ground_xy, float)
    return np.array([g[0], g[1], rim_height_m])


def rim_circle_3d(rim_ground_xy, rim_height_m: float = 3.048, rim_diameter_m: float = 0.4572, n: int = 60) -> np.ndarray:
    """Points on the horizontal rim circle in 3D court coords (for projection / ellipse
    cross-checks and Level-2 anchoring)."""
    center = rim_3d_center(rim_ground_xy, rim_height_m)
    r = rim_diameter_m / 2.0
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = np.stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t), np.full(n, center[2])], axis=1)
    return circ


# --------------------------------------------------------------------------- #
# On-disk annotation format
# --------------------------------------------------------------------------- #
@dataclass
class RimAnnotation:
    session_id: str
    image_size: tuple[int, int]                 # (width, height) px
    rim_ellipse: RimEllipse
    rim_ground_xy_m: tuple[float, float] = (0.0, 0.0)
    rim_height_m: float = 3.048
    rim_diameter_m: float = 0.4572
    net_region_px: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 below-net
    notes: str = ""

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "RimAnnotation":
        with open(path) as f:
            d = json.load(f)
        d["rim_ellipse"] = RimEllipse(**d["rim_ellipse"])
        d["image_size"] = tuple(d["image_size"])
        if d.get("net_region_px") is not None:
            d["net_region_px"] = tuple(d["net_region_px"])
        d["rim_ground_xy_m"] = tuple(d["rim_ground_xy_m"])
        return cls(**d)
