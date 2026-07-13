"""Constant-velocity Kalman filter — own implementation (plan §5.2, portfolio-relevant).

Theory: the Kalman filter is the closed-form Bayes filter under linear-Gaussian dynamics.
At a fixed camera and 60 fps-equivalent sampling, constant-velocity residuals are small,
so this is the right motion model for players (and for the ball *within a mode*, never as a
track-killing SORT filter). We implement a general linear KF and a CV constructor, and test
both against analytic cases (constant-velocity coasting, variance reduction, gating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class KalmanFilter:
    """General linear Kalman filter x_{k+1} = F x_k + w, z_k = H x_k + v."""

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F, float)
        self.H = np.asarray(H, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.asarray(P0, float)
        self.n = self.x.shape[0]

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z) -> np.ndarray:
        z = np.asarray(z, float).reshape(-1)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.n)
        # Joseph form for numerical stability of the covariance update.
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        return self.x

    def innovation(self, z) -> np.ndarray:
        z = np.asarray(z, float).reshape(-1)
        return z - self.H @ self.x

    def mahalanobis2(self, z) -> float:
        """Squared Mahalanobis distance of measurement z to the predicted measurement —
        the gating statistic for association."""
        y = self.innovation(z)
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.inv(S) @ y)


def cv_kalman_2d(x0_pos, dt: float = 1.0, *, process_std: float = 20.0,
                 meas_std: float = 3.0, init_vel=(0.0, 0.0), init_pos_var: float = 10.0,
                 init_vel_var: float = 100.0) -> KalmanFilter:
    """Constant-velocity KF for a 2D point. State = [x, y, vx, vy].

    process_std is the std of the (white-noise) acceleration in px/s^2; meas_std the
    localization noise in px. Q is the standard discrete white-noise-acceleration model."""
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    q = process_std ** 2
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    Q = q * np.array([
        [dt4 / 4, 0, dt3 / 2, 0],
        [0, dt4 / 4, 0, dt3 / 2],
        [dt3 / 2, 0, dt2, 0],
        [0, dt3 / 2, 0, dt2],
    ])
    R = (meas_std ** 2) * np.eye(2)
    x0 = np.array([x0_pos[0], x0_pos[1], init_vel[0], init_vel[1]], float)
    P0 = np.diag([init_pos_var, init_pos_var, init_vel_var, init_vel_var]).astype(float)
    return KalmanFilter(F, H, Q, R, x0, P0)


@dataclass
class TrackState:
    """Bookkeeping for a tracked target (used by the player tracker)."""

    track_id: int
    kf: KalmanFilter
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:4].copy()
