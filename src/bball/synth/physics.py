"""3D ballistic shot generator — the ground-truth physics behind every synthetic shot.

All constants are real basketball measurements (plan gate G4):
* g = 9.81 m/s^2
* rim height 3.048 m, rim diameter 0.4572 m (18 in) -> radius 0.2286 m
* ball diameter 0.24 m -> radius 0.12 m
* release speed 6-9 m/s, release angle 45-55 deg  (Phase-0 §10; NBA arc studies)
* release height 2.0-2.4 m (adult jump-shot release above the head)

A shot is built by *choosing its outcome*, then solving the launch that realizes it, so
every trajectory is exactly labelled (make/miss, miss direction/magnitude, zone). This is
what makes the FSM tests (M6) and the A5/A6 ablations possible. Airballs qualify as
attempts (review R3): a "short" miss reaches rim height well before the hoop.

The generator produces a `Shot`: sampled 3D positions (possession + flight), event
timestamps, and full ground-truth metadata. The renderer (render.py) projects it; the
event tests consume it directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# -- cited physical constants ------------------------------------------------
G = 9.81
RIM_HEIGHT_M = 3.048
RIM_DIAMETER_M = 0.4572
RIM_RADIUS_M = RIM_DIAMETER_M / 2.0
BALL_DIAMETER_M = 0.24
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0
RELEASE_SPEED_RANGE = (6.0, 9.0)
RELEASE_ANGLE_RANGE = (45.0, 55.0)
RELEASE_HEIGHT_RANGE = (2.0, 2.4)

Outcome = Literal["make", "miss"]
MissDir = Literal["short", "long", "left", "right", "short-left", "short-right",
                  "long-left", "long-right", "none"]


@dataclass
class Shot:
    """One fully-labelled synthetic shot."""

    t: np.ndarray                     # (T,) times in seconds, from possession start
    pos: np.ndarray                   # (T,3) ball centre in court metres
    fps: float
    outcome: Outcome
    miss_direction: MissDir
    miss_magnitude_m: float
    release_xy: np.ndarray            # shooter feet (last ground contact) court xy
    release_point: np.ndarray         # 3D ball release point
    hoop_ground_xy: np.ndarray
    release_speed: float
    release_angle_deg: float
    apex_height_m: float
    events: dict = field(default_factory=dict)   # release_t, apex_t, rim_arrival_t
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.pos.shape[0]


def solve_launch(
    release_point: np.ndarray,
    target_xy: np.ndarray,
    target_height: float,
    angle_deg: float,
) -> tuple[float, float, float]:
    """Solve the launch that carries the ball from `release_point` through the horizontal
    point `target_xy` at height `target_height`, at the given elevation angle.

    Returns (speed, azimuth_rad, time_of_flight). Raises ValueError if the angle is too
    shallow to reach the target height."""
    p = np.asarray(release_point, float)
    d = float(np.hypot(target_xy[0] - p[0], target_xy[1] - p[1]))
    dz = target_height - p[2]
    a = np.radians(angle_deg)
    denom = d * np.tan(a) - dz
    if denom <= 1e-6 or d < 1e-6:
        raise ValueError("launch angle too shallow (or zero distance) for this target")
    v = np.sqrt(G * d * d / (2.0 * np.cos(a) ** 2 * denom))
    beta = float(np.arctan2(target_xy[1] - p[1], target_xy[0] - p[0]))
    tof = d / (v * np.cos(a))
    return float(v), beta, float(tof)


def _ballistic(release_point, v, alpha, beta, times) -> np.ndarray:
    p = np.asarray(release_point, float)
    vx = v * np.cos(alpha) * np.cos(beta)
    vy = v * np.cos(alpha) * np.sin(beta)
    vz = v * np.sin(alpha)
    t = np.asarray(times, float)
    x = p[0] + vx * t
    y = p[1] + vy * t
    z = p[2] + vz * t - 0.5 * G * t * t
    return np.stack([x, y, z], axis=1)


_MISS_AXES = {
    "short": (-1, 0), "long": (1, 0), "left": (0, -1), "right": (0, 1),
    "short-left": (-1, -1), "short-right": (-1, 1),
    "long-left": (1, -1), "long-right": (1, 1), "none": (0, 0),
}


def generate_shot(
    *,
    release_xy,
    hoop_ground_xy=(0.0, 0.0),
    outcome: Outcome = "make",
    miss_direction: MissDir = "none",
    miss_magnitude_m: float = 0.6,
    release_angle_deg: float = 50.0,
    release_height_m: float = 2.2,
    rattle: bool = False,
    rattle_amplitude_m: float = 0.08,
    fps: float = 60.0,
    pre_release_s: float = 0.5,
    dribble: bool = False,
    post_rim_s: float = 0.5,
    seed: int | None = None,
) -> Shot:
    """Generate one labelled shot.

    The ball is aimed at `hoop_centre + offset`, where the offset is expressed in rim-local
    axes: +depth = past the hoop (long), -depth = short, +lateral = right. For a make the
    offset is ~0 (small jitter) and the ball funnels down through the net; for a miss the
    offset has magnitude `miss_magnitude_m` in the requested direction.
    """
    rng = np.random.default_rng(seed)
    release_xy = np.asarray(release_xy, float)
    hoop_ground_xy = np.asarray(hoop_ground_xy, float)
    release_point = np.array([release_xy[0], release_xy[1], release_height_m])

    # Rim-local axes: depth = shooter->hoop (horizontal), lateral = perpendicular.
    to_hoop = hoop_ground_xy - release_xy
    depth_hat = to_hoop / (np.linalg.norm(to_hoop) + 1e-9)
    lateral_hat = np.array([-depth_hat[1], depth_hat[0]])

    if outcome == "make":
        # Small in-rim jitter so makes are not all dead-centre.
        j = rng.normal(0, 0.35 * RIM_RADIUS_M, size=2)
        target_xy = hoop_ground_xy + j
        eff_miss_dir: MissDir = "none"
        eff_mag = float(np.hypot(*j))
    else:
        sd, sl = _MISS_AXES[miss_direction if miss_direction != "none" else "long"]
        offset = (sd * miss_magnitude_m) * depth_hat + (sl * miss_magnitude_m) * lateral_hat
        target_xy = hoop_ground_xy + offset
        eff_miss_dir = miss_direction if miss_direction != "none" else "long"
        eff_mag = miss_magnitude_m

    # A downward-curving parabola can only pass through a point that lies below its launch
    # ray, i.e. d*tan(alpha) > dz. Close shots to a higher rim therefore need a steeper
    # arc (a layup/floater goes up steeply) — clamp the sampled angle up to the minimum
    # feasible value with a margin. This keeps every sampled location realizable.
    d_target = float(np.hypot(*(target_xy - release_xy)))
    dz_target = RIM_HEIGHT_M - release_height_m
    min_angle = np.degrees(np.arctan2(max(dz_target, 0.0), max(d_target, 1e-3))) + 8.0
    eff_angle = float(np.clip(max(release_angle_deg, min_angle), release_angle_deg, 78.0))

    v, beta, tof = solve_launch(release_point, target_xy, RIM_HEIGHT_M, eff_angle)
    release_angle_deg = eff_angle

    dt = 1.0 / fps
    # -- possession phase (ball near hands; optional dribble oscillation) --
    n_pre = max(int(round(pre_release_s * fps)), 1)
    pre_t = np.arange(-n_pre, 0) * dt
    hold_z = release_height_m - 0.35  # ball held around chest/waist before the lift
    pre_pos = np.tile(np.array([release_xy[0], release_xy[1], hold_z]), (n_pre, 1))
    if dribble:
        # ~2 Hz vertical oscillation reaching near the floor (pull-up signature).
        osc = 1.0 - np.abs(np.sin(2 * np.pi * 2.0 * (pre_t - pre_t[0])))
        pre_pos[:, 2] = 0.15 + osc * (hold_z - 0.15)
    else:
        pre_pos[:, 2] += rng.normal(0, 0.01, size=n_pre)  # tiny hold jitter

    # -- flight phase to rim arrival --
    n_flight = max(int(round(tof * fps)), 2)
    fl_t = np.arange(0, n_flight + 1) * dt
    fl_pos = _ballistic(release_point, v, np.radians(release_angle_deg), beta, fl_t)
    apex_height = release_height_m + (v * np.sin(np.radians(release_angle_deg))) ** 2 / (2 * G)

    # -- terminal phase --
    post_t = np.arange(1, max(int(round(post_rim_s * fps)), 2) + 1) * dt
    rim_pos = fl_pos[-1].copy()
    if outcome == "make":
        # Funnel down through the net toward the hoop centre.
        drop = np.linspace(0, 1, len(post_t))
        term = np.zeros((len(post_t), 3))
        term[:, 0] = rim_pos[0] + (hoop_ground_xy[0] - rim_pos[0]) * drop
        term[:, 1] = rim_pos[1] + (hoop_ground_xy[1] - rim_pos[1]) * drop
        term[:, 2] = RIM_HEIGHT_M - drop * (RIM_HEIGHT_M - (RIM_HEIGHT_M - 0.9))
        if rattle:
            wob = rattle_amplitude_m * np.exp(-3 * drop) * np.sin(2 * np.pi * 6 * drop)
            term[:, 0] += wob
            term[:, 1] += wob * 0.5
    else:
        # Continue ballistic past the rim toward the floor (sails past / long-short/wide).
        vz_rim = v * np.sin(np.radians(release_angle_deg)) - G * tof
        term = np.zeros((len(post_t), 3))
        term[:, 0] = rim_pos[0] + v * np.cos(np.radians(release_angle_deg)) * np.cos(beta) * post_t
        term[:, 1] = rim_pos[1] + v * np.cos(np.radians(release_angle_deg)) * np.sin(beta) * post_t
        term[:, 2] = np.maximum(RIM_HEIGHT_M + vz_rim * post_t - 0.5 * G * post_t**2, BALL_RADIUS_M)

    # -- assemble --
    t_all = np.concatenate([pre_t, fl_t, tof + post_t])
    pos_all = np.vstack([pre_pos, fl_pos, term])
    t_all = t_all - t_all[0]  # start at 0
    release_t = pre_t.shape[0] * dt - dt  # last possession frame -> release boundary
    apex_t = release_t + (v * np.sin(np.radians(release_angle_deg))) / G
    rim_arrival_t = release_t + tof

    return Shot(
        t=t_all,
        pos=pos_all,
        fps=fps,
        outcome=outcome,
        miss_direction=eff_miss_dir,
        miss_magnitude_m=eff_mag,
        release_xy=release_xy,
        release_point=release_point,
        hoop_ground_xy=hoop_ground_xy,
        release_speed=v,
        release_angle_deg=release_angle_deg,
        apex_height_m=float(apex_height),
        events={"release_t": float(release_t), "apex_t": float(apex_t),
                "rim_arrival_t": float(rim_arrival_t)},
        meta={"dribble": dribble, "rattle": rattle, "target_xy": target_xy.tolist()},
    )


def sample_release_params(rng: np.random.Generator) -> dict:
    """Sample release angle/height uniformly within the cited ranges."""
    return {
        "release_angle_deg": float(rng.uniform(*RELEASE_ANGLE_RANGE)),
        "release_height_m": float(rng.uniform(*RELEASE_HEIGHT_RANGE)),
    }
