"""Rim-normalized shot FSM (plan §5.4, review R2).

POSSESSION -> RISING (release) -> DESCENDING (apex) -> RIM_INTERACTION -> {MADE|MISSED}
-> COOLDOWN. Predicates are expressed in **rim-normalized image coordinates**: the annotated
rim ellipse is the projective image of the rim circle, so "inside the rim" is a fraction of
its axes (radial_fraction), which absorbs camera geometry without unobservable 3D.

The MADE verdict integrates the **terminal state**, not the first crossing: the ball centre
passes downward through the rim-ellipse interior, is then seen (real or bridged) below the
rim for N consecutive frames, and does not reappear above without a new possession. This is
what makes rattle-in and shooter's-roll (cross, wobble, drop) resolve as makes while a
rim-out (enter, come back up) resolves as a miss. A margin score (interior depth x
confirmation x real-vs-bridged evidence) feeds calibration; the verdict is the terminal
state. Airballs are attempts (review R3): apex above the rim + motion toward the rim, but no
interior pass -> a miss, not a non-event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from bball.lift.rim_frame import RimEllipse


class State(Enum):
    POSSESSION = "possession"
    RISING = "rising"
    DESCENDING = "descending"
    RIM_INTERACTION = "rim_interaction"
    COOLDOWN = "cooldown"


@dataclass
class FSMConfig:
    make_fraction: float = 0.6         # lateral offset (major-axis fractions) allowed for a make
    attempt_fraction: float = 6.0      # lateral offset within which the flight counts as an attempt
    confirm_frames: int = 3            # consecutive net-region frames needed to confirm a make
    cooldown_frames: int = 20          # guards a single flight from double-counting
    net_width_factor: float = 1.3      # net box half-width in major-axis (a) units
    net_depth_factor: float = 6.0      # net box depth below the rim in minor-axis... see below
    smooth_win: int = 3                # velocity smoothing window


@dataclass
class ShotOutcome:
    outcome: str                       # "make" | "miss" | "none"
    margin_score: float                # raw margin (calibrate later)
    make_prob: float                   # uncalibrated sigmoid(margin)
    closest_fraction: float
    apex_above_rim: bool
    confirm_frames: int
    real_evidence_ratio: float
    release_idx: int = -1
    rim_idx: int = -1
    meta: dict = field(default_factory=dict)


def _rim_bounds(ell: RimEllipse):
    poly = ell.boundary_polyline(120)
    return {"top_y": float(poly[:, 1].min()), "bottom_y": float(poly[:, 1].max()),
            "cx": ell.cx, "cy": ell.cy}


def _smooth_velocity(xy: np.ndarray, win: int) -> np.ndarray:
    n = xy.shape[0]
    vel = np.zeros_like(xy)
    for i in range(1, n):
        j0 = max(0, i - win)
        seg = xy[j0:i + 1]
        if len(seg) >= 2 and not np.isnan(seg).any():
            vel[i] = (seg[-1] - seg[0]) / (len(seg) - 1)
        else:
            vel[i] = vel[i - 1]
    return vel


class ShotFSM:
    """Classifies one flight segment. Rim-normalized; consumes real or bridged positions."""

    def __init__(self, rim_ellipse: RimEllipse, config: FSMConfig | None = None):
        self.ell = rim_ellipse
        self.cfg = config or FSMConfig()
        self.bounds = _rim_bounds(rim_ellipse)

    def process_flight(self, ball_xy: np.ndarray, observed: np.ndarray | None = None) -> ShotOutcome:
        xy = np.asarray(ball_xy, float)
        n = xy.shape[0]
        if observed is None:
            observed = ~np.isnan(xy).any(axis=1)
        valid = ~np.isnan(xy).any(axis=1)
        norm = np.full((n, 2), np.nan)
        frac = np.full(n, np.nan)
        if valid.any():
            norm[valid] = self.ell.to_normalized(xy[valid])
            frac[valid] = np.sqrt((norm[valid] ** 2).sum(axis=1))
        vel = _smooth_velocity(xy, self.cfg.smooth_win)
        vy = vel[:, 1]                       # image-down velocity (>0 = moving down)
        cy = self.bounds["cy"]

        # Apex = highest image point (min y) among observed frames. The gate (apex above the
        # rim centre) still rejects a lob (which peaks below the rim -> apex_y > cy) while
        # admitting close shots whose apex only just clears the rim.
        ys = np.where(valid, xy[:, 1], np.inf)
        apex_idx = int(np.argmin(ys))
        apex_above_rim = bool(xy[apex_idx, 1] < cy) if valid.any() else False

        # Descending crossing of the rim level (image y passes cy going down): the rim-arrival
        # moment, robust to the depth ambiguity that fools a global closest-approach. If no
        # clean crossing exists (close shots whose image stays near/above cy), fall back to the
        # descending closest approach so the attempt is still counted.
        cross_idx = -1
        for i in range(1, n):
            if valid[i] and valid[i - 1] and xy[i - 1, 1] < cy <= xy[i, 1] and vy[i] > 0:
                cross_idx = i
                break
        if cross_idx < 0:
            desc = np.array([i for i in range(1, n) if valid[i] and vy[i] > 0 and i > apex_idx])
            if desc.size:
                cross_idx = int(desc[np.argmin(frac[desc])])
        if cross_idx < 1:
            frac_at = float(np.nanmin(frac)) if valid.any() else np.inf
            return ShotOutcome("none", -5.0, _sigmoid(-5.0), frac_at, apex_above_rim, 0, 0.0,
                               meta={"reason": "no_rim_crossing"})

        # The make gate is the lateral offset along the well-conditioned MAJOR axis at the
        # rim crossing, interpolated to the exact y = cy level (the ball drifts fast near the
        # rim, so the raw post-crossing frame under-reads the offset). At an off-axis camera
        # the depth axis (short/long) also projects onto the major axis, so this one robust
        # quantity rejects every miss direction; as the camera approaches the shooting lane the
        # depth projection shrinks and short/long becomes unobservable here — the A6 degradation.
        i = cross_idx
        denom = xy[i, 1] - xy[i - 1, 1]
        alpha = float(np.clip((cy - xy[i - 1, 1]) / denom, 0.0, 1.0)) if abs(denom) > 1e-6 else 0.0
        cross_pt = xy[i - 1] + alpha * (xy[i] - xy[i - 1])
        cross_norm = self.ell.to_normalized(cross_pt[None, :])[0]
        lateral = float(abs(cross_norm[0]))
        frac_cross = float(np.hypot(*cross_norm))

        is_attempt = apex_above_rim and lateral < self.cfg.attempt_fraction
        if not is_attempt:
            return ShotOutcome("none", -5.0, _sigmoid(-5.0), frac_cross, apex_above_rim, 0, 0.0,
                               meta={"reason": "not_an_attempt"})

        made, confirm, returned_above = self._terminal_make(xy, norm, frac, vy, cross_idx, cy, observed)
        lateral_ok = lateral < self.cfg.make_fraction
        made = made and lateral_ok
        s, e = max(cross_idx - 3, 0), min(cross_idx + self.cfg.confirm_frames + 1, n)
        real_ratio = float(observed[s:e].mean())

        # Margin is dominated by the net-dwell strength so it tracks the verdict: makes score
        # high, misses low. Central crossing adds a little; rim-out and lateral failures subtract.
        dwell_strength = min(confirm, 4 * self.cfg.confirm_frames) / self.cfg.confirm_frames - 1.0
        margin = (2.5 * dwell_strength
                  + 1.0 * (self.cfg.make_fraction - frac_cross)
                  - (3.0 if returned_above else 0.0)
                  - (0.0 if lateral_ok else 3.0))
        outcome = "make" if made else "miss"
        return ShotOutcome(
            outcome=outcome, margin_score=float(margin), make_prob=_sigmoid(margin),
            closest_fraction=float(frac_cross), apex_above_rim=apex_above_rim,
            confirm_frames=int(confirm), real_evidence_ratio=real_ratio,
            rim_idx=cross_idx, meta={"returned_above": returned_above, "lateral": float(lateral)})

    def _terminal_make(self, xy, norm, frac, vy, cross_idx, cy, observed):
        """After the descending rim crossing, a make requires the ball to enter the net region
        (directly below the rim centre) for confirm_frames and not pop back above the rim
        *after* settling. A frame counts toward the dwell when it is observed below the rim
        near centre OR when the ball is lost to occlusion below the rim: a ball that crosses
        down through the interior and vanishes into the net (without reappearing above) is the
        make signal — the occlusion gap is evidence, not absence of it. An OBSERVED reappearance
        above the rim after settling is a rim-out. Order matters: a shooter's roll can ride
        above the rim briefly BEFORE settling (still a make)."""
        n = xy.shape[0]
        net_w = self.cfg.net_width_factor
        confirm = 0
        best_run = 0
        dwell_at = -1
        for i in range(cross_idx, n):
            lost = (not observed[i]) or np.isnan(xy[i]).any()
            if lost:
                confirm += 1                      # ball swallowed by the net / occluded below
            elif abs(norm[i, 0]) < net_w and xy[i, 1] > cy:
                confirm += 1
            else:
                confirm = 0
            best_run = max(best_run, confirm)
            if confirm >= self.cfg.confirm_frames and dwell_at < 0:
                dwell_at = i
        returned_above = False
        if dwell_at >= 0:
            for i in range(dwell_at, n):
                if observed[i] and not np.isnan(xy[i]).any() and xy[i, 1] < self.bounds["top_y"]:
                    returned_above = True         # popped out after settling -> rim-out
                    break
        made = (dwell_at >= 0) and not returned_above
        return made, best_run, returned_above


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# --------------------------------------------------------------------------- #
# Streaming orchestrator: segment flights, apply cooldown, handle put-backs
# --------------------------------------------------------------------------- #
@dataclass
class ShotEvent:
    outcome: str
    make_prob: float
    margin_score: float
    release_t: float
    rim_t: float
    flight_span: tuple[int, int]
    detail: ShotOutcome


def run_fsm_stream(ball_xy: np.ndarray, times: np.ndarray, rim_ellipse: RimEllipse,
                   *, config: FSMConfig | None = None,
                   segments: list[tuple[int, int]] | None = None) -> list[ShotEvent]:
    """Run the FSM over a continuous stream. `segments` are flight (start,end) spans from
    the release detector (bball.events.release); if omitted, the whole stream is one flight.
    COOLDOWN between accepted events prevents a single flight (or its rebound) from being
    double-counted; a genuinely new flight (put-back) after cooldown is a new attempt."""
    cfg = config or FSMConfig()
    fsm = ShotFSM(rim_ellipse, cfg)
    if segments is None:
        segments = [(0, len(ball_xy))]
    events: list[ShotEvent] = []
    last_rim_frame = -10 ** 9
    for (s, e) in segments:
        out = fsm.process_flight(ball_xy[s:e])
        if out.outcome == "none":
            continue
        rim_global = s + max(out.rim_idx, 0)
        if rim_global - last_rim_frame < cfg.cooldown_frames:
            continue   # within cooldown of the previous event -> same flight / rebound
        last_rim_frame = rim_global
        events.append(ShotEvent(
            outcome=out.outcome, make_prob=out.make_prob, margin_score=out.margin_score,
            release_t=float(times[s]), rim_t=float(times[min(rim_global, len(times) - 1)]),
            flight_span=(s, e), detail=out))
    return events
