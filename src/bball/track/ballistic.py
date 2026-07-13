"""Two-level trajectory model (plan §5.2, review R1).

A homography cannot lift an airborne ball, and a 3D parabola does not project to an exact
image parabola. So each level is used only for what it can support:

* **Level 1 — image-space quadratic** over short windows: assumption-light, robust. Used
  for association gating and occlusion bridging (this is also what Apple's
  VNDetectTrajectoriesRequest fits). Multi-ball robustness falls out of the gate: a second
  ball violating the fit is rejected.
* **Level 2 — constrained 3D reconstruction**: a parabola in a vertical plane with g fixed
  at 9.81, anchored by the rim's 3D position, the shooter's ground point + a release-height
  band, and the ball's known diameter as a noisy depth cue. Free params are the release
  height and launch velocity; the fit reports residuals and a confidence that gates every
  metric output (T5 miss direction, arc summaries). Its depth axis is well-constrained only
  when the camera sees it — which is exactly the azimuth dependence A6 measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from bball.detect.interfaces import BallCandidate

G = 9.81


# --------------------------------------------------------------------------- #
# Level 1 — image-space quadratic
# --------------------------------------------------------------------------- #
def fit_quadratic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares quadratic y = a t^2 + b t + c; returns [a, b, c]."""
    A = np.stack([t * t, t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


@dataclass
class Level1Fit:
    ax: np.ndarray  # coeffs for x(t)
    ay: np.ndarray  # coeffs for y(t)
    t0: float

    def predict(self, t) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, float)) - self.t0
        x = self.ax[0] * tt * tt + self.ax[1] * tt + self.ax[2]
        y = self.ay[0] * tt * tt + self.ay[1] * tt + self.ay[2]
        return np.stack([x, y], axis=1)


def fit_level1(times: np.ndarray, xy: np.ndarray) -> Level1Fit:
    t = np.asarray(times, float)
    xy = np.asarray(xy, float)
    t0 = t[0]
    return Level1Fit(ax=fit_quadratic(t - t0, xy[:, 0]), ay=fit_quadratic(t - t0, xy[:, 1]), t0=t0)


@dataclass
class BridgeResult:
    xy: np.ndarray            # (N,2) filled positions (NaN where unfillable)
    observed: np.ndarray      # (N,) bool — real detection accepted this frame
    bridged: np.ndarray       # (N,) bool — position came from a fit prediction
    method: str
    gaps: list = field(default_factory=list)   # (start, length) of bridged runs

    @property
    def completeness(self) -> float:
        return float(np.mean(~np.isnan(self.xy).any(axis=1)))


def bridge_trajectory(
    candidates: list[BallCandidate],
    times: np.ndarray,
    *,
    method: str = "l1",
    window: int = 12,
    base_gate_px: float = 40.0,
    gate_growth_px: float = 12.0,
    min_fit: int = 4,
    camera=None,
    reconstruct_kwargs: dict | None = None,
) -> BridgeResult:
    """Fill missing/occluded frames with fitted predictions and reject gate-violating
    candidates. `method`: 'off' (no bridging), 'l1' (image-space quadratic), 'l2'
    (3D-anchored: project the reconstructed parabola through the gap)."""
    n = len(candidates)
    t = np.asarray(times, float)
    xy = np.full((n, 2), np.nan)
    observed = np.zeros(n, bool)
    bridged = np.zeros(n, bool)

    win_t: list[float] = []
    win_xy: list[np.ndarray] = []
    n_missed = 0

    # Precompute an L2 image-space predictor if requested.
    l2_predictor = None
    if method == "l2" and camera is not None:
        l2_predictor = _build_l2_image_predictor(candidates, t, camera, reconstruct_kwargs or {})

    for i in range(n):
        cand = candidates[i]
        pred = None
        gate = base_gate_px + gate_growth_px * n_missed
        if method == "l2" and l2_predictor is not None:
            pred = l2_predictor(t[i])
        elif method in ("l1", "l2") and len(win_t) >= min_fit:
            fit = fit_level1(np.array(win_t), np.array(win_xy))
            pred = fit.predict(t[i])[0]

        if cand.observed:
            if method == "off" or pred is None or np.hypot(*(cand.xy - pred)) <= gate:
                xy[i] = cand.xy
                observed[i] = True
                win_t.append(t[i]); win_xy.append(cand.xy)
                if len(win_t) > window:
                    win_t.pop(0); win_xy.pop(0)
                n_missed = 0
            else:  # gate violation: physics-inconsistent candidate (e.g. a second ball)
                if method != "off" and pred is not None:
                    xy[i] = pred; bridged[i] = True
                n_missed += 1
        else:
            if method != "off" and pred is not None:
                xy[i] = pred; bridged[i] = True
            n_missed += 1

    return BridgeResult(xy=xy, observed=observed, bridged=bridged, method=method,
                        gaps=_runs(bridged))


def _runs(mask: np.ndarray) -> list:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - start)); start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


# --------------------------------------------------------------------------- #
# Level 2 — constrained 3D reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory3D:
    times: np.ndarray
    positions: np.ndarray     # (N,3) court metres
    confidence: float
    rms_reproj_px: float
    params: dict = field(default_factory=dict)


def _parabola3d(p0, v0, times):
    t = np.asarray(times, float)[:, None]
    acc = np.array([0.0, 0.0, -G])
    return p0[None, :] + v0[None, :] * t + 0.5 * acc[None, :] * t * t


def reconstruct_flight_3d(
    image_obs: np.ndarray,           # (N,2) observed image points (NaN where missing)
    times: np.ndarray,               # (N,) seconds
    camera,
    *,
    shooter_feet_xy,
    rim_center_3d,
    release_height_band=(2.0, 2.4),
    ball_radius_px: np.ndarray | None = None,
    ball_diameter_m: float = 0.24,
    rim_weight: float = 8.0,
    depth_weight: float = 2.0,
) -> Trajectory3D:
    """Fit a vertical-plane parabola to image observations, anchored by the shooter ground
    point (release xy), a release-height band (z0), and the rim 3D position. Free params:
    [z0, vx, vy, vz]. Confidence falls with reprojection RMS and ill-conditioning."""
    obs = np.asarray(image_obs, float)
    t = np.asarray(times, float)
    valid = ~np.isnan(obs).any(axis=1)
    tv = t[valid]
    ov = obs[valid]
    feet = np.asarray(shooter_feet_xy, float)
    rim = np.asarray(rim_center_3d, float)
    f = camera.K[0, 0]

    # Time origin: the parabola P0 + V0 t - 0.5 g t^2 has P0 at t=0, so measure time from
    # the first observed flight frame (~release), not from the clip start.
    t0 = float(tv[0])
    tv0 = tv - t0

    # Initial guess: aim from feet@release_band_mid toward the rim over the observed span.
    z0_guess = float(np.mean(release_height_band))
    span = max(tv[-1] - tv[0], 1e-2)
    horiz = rim[:2] - feet
    v_guess = np.array([horiz[0] / span, horiz[1] / span,
                        (rim[2] - z0_guess) / span + 0.5 * G * span])
    p_init = np.array([z0_guess, v_guess[0], v_guess[1], v_guess[2]])

    # The rim and ball-size priors only correct what reprojection cannot see: the depth
    # axis. Gate them by (1 - depth_observability) so a side-on camera (reprojection sees
    # everything) fits reprojection-pure and accurate, while an end-on camera leans on the
    # priors for a necessarily noisier depth estimate. This *is* the A6 azimuth dependence.
    depth_obs = _depth_observability(camera, feet, rim)
    w_rim = rim_weight * (1.0 - depth_obs)
    w_depth = depth_weight * (1.0 - depth_obs)

    def unpack(p):
        p0 = np.array([feet[0], feet[1], p[0]])
        v0 = np.array([p[1], p[2], p[3]])
        return p0, v0

    def residuals(p):
        p0, v0 = unpack(p)
        pos = _parabola3d(p0, v0, tv0)
        proj = camera.project(pos)
        res = (proj - ov).ravel()
        # Soft rim-pass: the point nearest rim height on descent should be near the rim.
        z = pos[:, 2]
        if w_rim > 1e-6 and np.any(z >= rim[2]):
            k = int(np.argmin(np.abs(z - rim[2])))
            res = np.concatenate([res, w_rim * (pos[k, :2] - rim[:2])])
        # Depth-from-size cue: apparent radius ~ f*r/depth.
        if w_depth > 1e-6 and ball_radius_px is not None:
            rv = np.asarray(ball_radius_px, float)[valid]
            depth = (camera.R @ (pos - camera.position).T)[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                pred_r = f * (ball_diameter_m / 2.0) / depth
            ok = np.isfinite(pred_r) & np.isfinite(rv) & (rv > 0)
            if ok.any():
                res = np.concatenate([res, w_depth * (pred_r[ok] - rv[ok])])
        return res

    sol = least_squares(residuals, p_init, method="lm", max_nfev=300)
    p0, v0 = unpack(sol.x)
    pos = _parabola3d(p0, v0, t - t0)
    proj = camera.project(pos[valid])
    rms = float(np.sqrt(np.nanmean(((proj - ov) ** 2).sum(axis=1))))

    # Confidence: reprojection quality x depth observability (Jacobian conditioning along
    # the shooter->hoop depth axis). End-on cameras make the depth axis unobservable.
    conf = float(np.exp(-rms / 8.0) * depth_obs)
    return Trajectory3D(times=t, positions=pos, confidence=np.clip(conf, 0.0, 1.0),
                        rms_reproj_px=rms, params={"z0": float(sol.x[0]), "t0": t0,
                                                   "v0": v0.tolist(), "depth_observability": depth_obs})


def _depth_observability(camera, shooter_feet_xy, rim_center_3d) -> float:
    """How much a metre of motion along the shooter->hoop depth axis moves the image (px),
    normalized — proxy for how well the single camera constrains short/long. ~1 side-on,
    ->0 end-on."""
    feet = np.asarray(shooter_feet_xy, float)
    rim = np.asarray(rim_center_3d, float)
    depth_hat = np.array([rim[0] - feet[0], rim[1] - feet[1], 0.0])
    depth_hat /= (np.linalg.norm(depth_hat) + 1e-9)
    lateral_hat = np.array([-depth_hat[1], depth_hat[0], 0.0])
    base = np.array([rim[0], rim[1], rim[2]])
    p_img = camera.project(base[None, :])[0]
    d_img = camera.project((base + 0.5 * depth_hat)[None, :])[0]
    l_img = camera.project((base + 0.5 * lateral_hat)[None, :])[0]
    depth_px = np.hypot(*(d_img - p_img))
    lat_px = np.hypot(*(l_img - p_img))
    return float(depth_px / (depth_px + lat_px + 1e-9))


def _build_l2_image_predictor(candidates, times, camera, reconstruct_kwargs):
    """Fit L2 on observed flight points and return a function t -> predicted image xy."""
    obs = np.array([c.xy if c.observed else [np.nan, np.nan] for c in candidates], float)
    rad = np.array([c.radius_px if c.observed else np.nan for c in candidates], float)
    if (~np.isnan(obs).any(axis=1)).sum() < 5:
        return None
    try:
        traj = reconstruct_flight_3d(obs, times, camera, ball_radius_px=rad, **reconstruct_kwargs)
    except Exception:
        return None
    p0 = np.array([reconstruct_kwargs["shooter_feet_xy"][0],
                   reconstruct_kwargs["shooter_feet_xy"][1], traj.params["z0"]])
    v0 = np.array(traj.params["v0"])
    t0 = traj.params["t0"]

    def predict(t):
        pos = _parabola3d(p0, v0, np.array([t - t0]))
        return camera.project(pos)[0]

    return predict


def miss_vector_rim_local(traj3d: Trajectory3D, rim_center_3d, shooter_feet_xy) -> dict:
    """Decompose the ball's rim-plane crossing into depth (short/long) and lateral
    (left/right) with per-axis confidence."""
    rim = np.asarray(rim_center_3d, float)
    feet = np.asarray(shooter_feet_xy, float)
    pos = traj3d.positions
    z = pos[:, 2]
    # crossing of rim height on descent
    descending = np.gradient(z) < 0
    near = np.abs(z - rim[2]) + (~descending) * 1e3
    k = int(np.argmin(near))
    crossing = pos[k]
    depth_hat = np.array([rim[0] - feet[0], rim[1] - feet[1]])
    depth_hat = depth_hat / (np.linalg.norm(depth_hat) + 1e-9)
    lateral_hat = np.array([-depth_hat[1], depth_hat[0]])
    rel = crossing[:2] - rim[:2]
    depth = float(rel @ depth_hat)      # +long, -short
    lateral = float(rel @ lateral_hat)  # +right, -left (in shooter frame)
    depth_obs = traj3d.params.get("depth_observability", 0.5)
    return {
        "depth_m": depth, "lateral_m": lateral,
        "short_long": ("long" if depth > 0 else "short"),
        "left_right": ("right" if lateral > 0 else "left"),
        "lateral_confidence": float(np.clip(traj3d.confidence / max(depth_obs, 1e-3), 0, 1)) if depth_obs > 0 else traj3d.confidence,
        "depth_confidence": float(np.clip(traj3d.confidence, 0, 1)),
        "crossing_xy": crossing[:2].tolist(),
    }
