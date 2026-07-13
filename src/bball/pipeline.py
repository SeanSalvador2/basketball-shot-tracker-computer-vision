"""End-to-end pipeline: DETECT -> TRACK -> LIFT -> CLASSIFY on a clip/session.

This is the one-command spine (plan §5.0, gate G1). It processes a shot's rendered frames
with the classical bg-sub detector (no weights needed — Stage-A honest), bridges the ball
track through occlusion, runs the rim-normalized FSM, and lifts the shooter's ground position
to court coordinates via the session homography. A session-level report aggregates events,
the shot chart, and calibrated make probabilities.

Detection here uses bg-sub so the demo runs with zero downloaded weights; swapping in the
torchvision or a fine-tuned detector is a one-line change (same BallCandidate contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bball.detect.bgsub import BgSubBallDetector, BgSubConfig
from bball.detect.interfaces import BallCandidate
from bball.events.calibration import PlattScaler
from bball.events.fsm import FSMConfig, ShotFSM, run_fsm_stream
from bball.events.release import FlightSegmenter
from bball.lift.court_model import CourtSpec, classify_with_band
from bball.lift.homography import apply_homography
from bball.lift.rim_frame import RimEllipse


@dataclass
class ShotResult:
    outcome: str
    make_prob: float
    margin: float
    release_t: float
    rim_t: float
    court_xy: tuple[float, float] | None = None
    zone: str | None = None
    on_line: bool = False


@dataclass
class SessionReport:
    shots: list[ShotResult] = field(default_factory=list)

    @property
    def n_attempts(self) -> int:
        return len(self.shots)

    @property
    def n_makes(self) -> int:
        return sum(1 for s in self.shots if s.outcome == "make")

    def fg_pct(self) -> float:
        return self.n_makes / max(self.n_attempts, 1)

    def shot_chart_data(self) -> list[dict]:
        return [{"xy": s.court_xy, "outcome": s.outcome, "zone": s.zone}
                for s in self.shots if s.court_xy is not None]

    def summary(self) -> dict:
        from collections import Counter

        zones = Counter(s.zone for s in self.shots if s.zone)
        zone_fg = {}
        for z in zones:
            att = [s for s in self.shots if s.zone == z]
            zone_fg[z] = {"attempts": len(att), "makes": sum(1 for s in att if s.outcome == "make")}
        return {"attempts": self.n_attempts, "makes": self.n_makes, "fg_pct": round(self.fg_pct(), 3),
                "by_zone": zone_fg}


def detect_ball_bgsub(frames: list[np.ndarray], scale: float, *, config: BgSubConfig | None = None) -> list[BallCandidate]:
    """Run bg-sub over a clip; return one native-pixel BallCandidate per frame (best blob)."""
    det = BgSubBallDetector(config or BgSubConfig(min_area=4, max_area=4000))
    per_frame = det.process_stream(frames, temporal=True)
    out = []
    for i, cands in enumerate(per_frame):
        valid = [c for c in cands if c.xy is not None]
        if valid:
            best = max(valid, key=lambda c: c.score)          # most ball-like blob
            out.append(BallCandidate(i, best.xy / scale, best.score, best.radius_px / scale, source="bgsub"))
        else:
            out.append(BallCandidate(i, None))
    return out


def track_and_classify(candidates: list[BallCandidate], times: np.ndarray, rim_ellipse: RimEllipse,
                       *, fsm_config: FSMConfig | None = None, bridge_method: str = "l1") -> list:
    """Bridge the ball track, segment flights, run the FSM. Returns the FSM ShotEvents."""
    from bball.track.ballistic import bridge_trajectory

    br = bridge_trajectory(candidates, times, method=bridge_method)
    segs = FlightSegmenter().segment(br.xy, times)
    if not segs:
        segs = [(0, len(br.xy))]
    return run_fsm_stream(br.xy, times, rim_ellipse, config=fsm_config, segments=segs)


def lift_shooter(feet_xy_img: np.ndarray, homography_img_to_court: np.ndarray, court: CourtSpec,
                 *, on_line_band_m: float = 0.15) -> dict:
    """Map a shooter's image foot position to court coordinates + zone."""
    court_xy = apply_homography(homography_img_to_court, np.atleast_2d(feet_xy_img))[0]
    z = classify_with_band(court, court_xy[0], court_xy[1], on_line_band_m=on_line_band_m)
    return {"court_xy": (float(court_xy[0]), float(court_xy[1])), "zone": z["zone"], "on_line": z["on_line"]}


def build_session_report(shot_events: list, shooter_court_positions: list, zones: list,
                         calibrator: PlattScaler | None = None) -> SessionReport:
    """Assemble a session report from per-shot FSM events + lifted shooter positions."""
    report = SessionReport()
    for ev, court_xy, zinfo in zip(shot_events, shooter_court_positions, zones):
        prob = calibrator.predict(np.array([ev.margin_score]))[0] if calibrator else ev.make_prob
        report.shots.append(ShotResult(
            outcome=ev.outcome, make_prob=float(prob), margin=ev.margin_score,
            release_t=ev.release_t, rim_t=ev.rim_t, court_xy=court_xy,
            zone=zinfo.get("zone") if zinfo else None, on_line=zinfo.get("on_line", False) if zinfo else False))
    return report
