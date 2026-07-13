"""M6 event tests — the credibility core of the project.

Scripted synthetic scenarios each assert the correct event: clean make, clean misses
(short/long/left/right), rattle-in, shooter's roll, rim-out, rebound put-back, lob-pass
negative, multi-ball distractor. Plus release detection, last-ground-contact, flight
segmentation, miss-direction decomposition, and probability calibration.

The FSM tests use a tripod placement (1.5 m, ~50 deg wing) where the rim ellipse is
well-formed and short/long is observable; A6 quantifies degradation off this placement.
"""
from __future__ import annotations

import numpy as np
import pytest

from bball.detect.interfaces import BallCandidate
from bball.events.calibration import (
    PlattScaler,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from bball.events.fsm import ShotFSM, run_fsm_stream
from bball.events.miss_direction import decompose_miss
from bball.events.release import (
    FlightSegmenter,
    detect_release_fallback,
    detect_release_pose,
    last_ground_contact_frame,
)
from bball.lift.court_model import get_court
from bball.lift.rim_frame import rim_3d_center
from bball.synth.camera import apparent_ball_radius_px, make_camera, project_trajectory
from bball.synth.physics import RIM_HEIGHT_M, generate_shot
from bball.synth.render import DetectionNoiseModel, compute_rim_image_geometry, occlusion_fraction
from bball.synth.scenarios import generate_lob_pass
from bball.track.ballistic import bridge_trajectory, reconstruct_flight_3d

DT = 1.0 / 60.0


def _cam():
    return make_camera(azimuth_deg=50, height_m=1.5, distance_m=9.0)


def _fsm(cam):
    return ShotFSM(compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse)


def _outcome(cam, fsm, shot):
    return fsm.process_flight(cam.project(shot.pos)).outcome


def _descend_to(z_end, n=34, x=0.03, y=0.0, z0=4.0):
    z = z0 - 0.5 * 9.81 * (np.arange(n) * DT) ** 2
    z = z[z >= z_end]
    return np.stack([np.full_like(z, x), np.full_like(z, y), z], axis=1)


# --------------------------------------------------------------------------- #
# Core make/miss scenarios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw,expected", [
    (dict(outcome="make"), "make"),
    (dict(outcome="miss", miss_direction="short", miss_magnitude_m=1.0), "miss"),
    (dict(outcome="miss", miss_direction="long", miss_magnitude_m=0.7), "miss"),
    (dict(outcome="miss", miss_direction="left", miss_magnitude_m=0.6), "miss"),
    (dict(outcome="miss", miss_direction="right", miss_magnitude_m=0.6), "miss"),
    (dict(outcome="make", rattle=True), "make"),
])
def test_clean_scenarios(kw, expected):
    cam = _cam()
    fsm = _fsm(cam)
    shot = generate_shot(release_xy=(1.5, 6.8), hoop_ground_xy=(0, 0), fps=60, seed=2, **kw)
    assert _outcome(cam, fsm, shot) == expected


def test_make_is_confident_miss_is_not():
    cam = _cam()
    fsm = _fsm(cam)
    mk = fsm.process_flight(cam.project(generate_shot(release_xy=(1.5, 6.8), outcome="make", seed=2).pos))
    ms = fsm.process_flight(cam.project(generate_shot(release_xy=(1.5, 6.8), outcome="miss",
                                                      miss_direction="left", miss_magnitude_m=0.6, seed=2).pos))
    assert mk.make_prob > 0.7 and ms.make_prob < 0.3


def test_shooters_roll_is_a_make():
    """Ball rides the rim (briefly above it) then drops in — still a make."""
    cam = _cam()
    fsm = _fsm(cam)
    d = _descend_to(3.0)
    tr = np.arange(14) * DT
    roll = np.stack([0.15 * np.cos(2 * np.pi * 1.5 * tr), 0.15 * np.sin(2 * np.pi * 1.5 * tr),
                     3.048 + 0.06 * np.sin(2 * np.pi * 1.5 * tr)], axis=1)
    drop = np.stack([np.zeros(10), np.zeros(10), np.linspace(3.0, 2.1, 10)], axis=1)
    traj = np.vstack([d, roll, drop])
    assert fsm.process_flight(cam.project(traj)).outcome == "make"


def test_rim_out_is_a_miss():
    """Ball dips into the rim then pops back out and away — a miss."""
    cam = _cam()
    fsm = _fsm(cam)
    d = _descend_to(2.80)
    tt = np.arange(1, 26) * DT
    bx = d[-1, 0] + 0.9 * (tt / tt[-1]); by = d[-1, 1] - 0.6 * (tt / tt[-1])
    bz = d[-1, 2] + 3.4 * tt - 0.5 * 9.81 * tt ** 2
    ro = np.vstack([d, np.stack([bx, by, np.maximum(bz, 0.12)], axis=1)])
    assert fsm.process_flight(cam.project(ro)).outcome == "miss"


def test_lob_pass_is_not_an_attempt():
    cam = _cam()
    fsm = _fsm(cam)
    lob = generate_lob_pass(get_court("nba"), fps=60, seed=1)
    out = fsm.process_flight(cam.project(lob.pos))
    assert out.outcome == "none"


def test_multiball_distractor_does_not_break_make():
    """A second ball far from the flight is gated out by the trajectory bridging, so the
    FSM still sees a clean make."""
    cam = _cam()
    fsm = _fsm(cam)
    shot = generate_shot(release_xy=(1.0, 6.5), outcome="make", fps=60, seed=5)
    img = project_trajectory(cam, shot.pos)
    rad = apparent_ball_radius_px(cam, shot.pos)
    cands = []
    for i in range(shot.n_frames):
        xy = None if np.isnan(img[i]).any() else img[i].copy()
        cands.append(BallCandidate(i, xy, 0.9, float(rad[i]) if not np.isnan(rad[i]) else 3.0))
    for i in range(10, 40, 2):                       # inject a rolling second ball
        base = img[i] if not np.isnan(img[i]).any() else np.array([300.0, 700.0])
        cands[i] = BallCandidate(i, base + np.array([260.0, 190.0]), 0.85, 6.0)
    br = bridge_trajectory(cands, shot.t, method="l1")
    assert fsm.process_flight(br.xy, br.observed).outcome == "make"


def test_make_survives_rim_occlusion_via_bridging():
    """The ball vanishes into the net (occlusion-driven detection gap); Level-1 bridging
    fills it and the FSM still confirms the make."""
    cam = _cam()
    rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
    fsm = ShotFSM(rim_geom.ellipse)
    shot = generate_shot(release_xy=(1.0, 6.5), outcome="make", fps=60, seed=9)
    img = project_trajectory(cam, shot.pos)
    rad = apparent_ball_radius_px(cam, shot.pos)
    occl = np.array([occlusion_fraction(None if np.isnan(img[i]).any() else img[i],
                                        float(rad[i]) if not np.isnan(rad[i]) else 3.0, rim_geom)
                     for i in range(shot.n_frames)])
    stream = DetectionNoiseModel().stream(img, rad, occl, np.random.default_rng(0))
    assert sum(1 for c in stream if not c.observed) >= 1     # there really are gaps
    br = bridge_trajectory(stream, shot.t, method="l1")
    assert fsm.process_flight(br.xy, br.observed).outcome == "make"


# --------------------------------------------------------------------------- #
# Streaming: rebound put-back = two attempts, not a double-count
# --------------------------------------------------------------------------- #
def test_rebound_putback_counts_two_attempts():
    cam = _cam()
    ell = compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse
    miss = generate_shot(release_xy=(2.0, 6.0), outcome="miss", miss_direction="long",
                         miss_magnitude_m=0.7, fps=60, seed=3, post_rim_s=0.6)
    make2 = generate_shot(release_xy=(0.5, 2.5), outcome="make", fps=60, seed=4, pre_release_s=0.5)
    gap = np.tile(make2.pos[0], (20, 1))
    stream = np.vstack([miss.pos, gap, make2.pos])
    img = cam.project(stream)
    times = np.arange(len(stream)) * DT
    segs = FlightSegmenter().segment(img, times)
    events = run_fsm_stream(img, times, ell, segments=segs)
    outcomes = [e.outcome for e in events]
    assert outcomes == ["miss", "make"]              # two attempts, correct order, no double-count


def test_cooldown_prevents_double_count_of_single_flight():
    cam = _cam()
    ell = compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse
    make = generate_shot(release_xy=(1.0, 6.5), outcome="make", fps=60, seed=1)
    img = cam.project(make.pos)
    times = make.t
    # Feed the same flight twice as two overlapping segments; cooldown must collapse them.
    events = run_fsm_stream(img, times, ell, segments=[(0, len(img)), (2, len(img))])
    assert len(events) == 1


# --------------------------------------------------------------------------- #
# Release / feet
# --------------------------------------------------------------------------- #
def test_release_fallback_near_true_release():
    cam = _cam()
    shot = generate_shot(release_xy=(1.5, 6.5), outcome="make", fps=60, pre_release_s=0.5, seed=1)
    img = cam.project(shot.pos)
    idx = detect_release_fallback(img, rise_vel_px=1.0)
    true_idx = int(round(shot.events["release_t"] * 60))
    assert idx >= 0 and abs(idx - true_idx) <= 4          # within +-4 frames


def test_release_pose_on_separation():
    n = 40
    ball = np.zeros((n, 2))
    wrist = np.zeros((n, 2))
    for i in range(n):
        wrist[i] = [100, 300 - i * 0.5]
        ball[i] = [100, 300 - (i * 0.5 if i < 15 else 15 * 0.5 + (i - 15) * 6)]  # ball launches at 15
    idx = detect_release_pose(ball, wrist, ball_radius_px=5.0, sep_mult=1.5)
    assert 14 <= idx <= 18


def test_last_ground_contact_before_release():
    feet_h = np.array([0.0, 0.0, 0.0, 0.05, 0.2, 0.5, 0.9])   # lifts off at frame 4
    assert last_ground_contact_frame(feet_h, release_idx=6, ground_thresh_m=0.08) == 3


def test_flight_segmenter_finds_two_shots():
    cam = _cam()
    s1 = generate_shot(release_xy=(2.0, 6.0), outcome="miss", miss_direction="left", fps=60, seed=1)
    s2 = generate_shot(release_xy=(-1.0, 5.0), outcome="make", fps=60, seed=2)
    stream = np.vstack([s1.pos, np.tile(s2.pos[0], (15, 1)), s2.pos])
    img = cam.project(stream)
    segs = FlightSegmenter().segment(img)
    assert len(segs) >= 2


# --------------------------------------------------------------------------- #
# Miss direction
# --------------------------------------------------------------------------- #
def test_miss_direction_lateral_correct_side():
    cam = make_camera(azimuth_deg=90, height_m=3.0, distance_m=10.0)  # side-on: lateral observable
    for side, expect in [("left", "left"), ("right", "right")]:
        shot = generate_shot(release_xy=(0.0, 6.5), outcome="miss", miss_direction=side,
                             miss_magnitude_m=0.6, seed=8)
        mask = (shot.t >= shot.events["release_t"]) & (shot.t <= shot.events["rim_arrival_t"] + 0.05)
        pos = shot.pos[mask]; t = shot.t[mask]
        traj = reconstruct_flight_3d(cam.project(pos), t, cam, shooter_feet_xy=shot.release_xy,
                                     rim_center_3d=rim_3d_center((0.0, 0.0)),
                                     ball_radius_px=apparent_ball_radius_px(cam, pos))
        res = decompose_miss(traj, rim_3d_center((0.0, 0.0)), shot.release_xy)
        assert res.left_right.label == expect
        assert res.left_right.shown


def test_short_long_hidden_when_confidence_low():
    """End-on camera: depth confidence collapses, so short/long is hidden not guessed."""
    cam = make_camera(azimuth_deg=8, height_m=3.0, distance_m=10.0)  # near the shooting lane
    shot = generate_shot(release_xy=(0.0, 7.0), outcome="miss", miss_direction="long",
                         miss_magnitude_m=0.7, seed=8)
    mask = (shot.t >= shot.events["release_t"]) & (shot.t <= shot.events["rim_arrival_t"] + 0.05)
    pos = shot.pos[mask]; t = shot.t[mask]
    traj = reconstruct_flight_3d(cam.project(pos), t, cam, shooter_feet_xy=shot.release_xy,
                                 rim_center_3d=rim_3d_center((0.0, 0.0)),
                                 ball_radius_px=apparent_ball_radius_px(cam, pos))
    res = decompose_miss(traj, rim_3d_center((0.0, 0.0)), shot.release_xy, min_depth_conf=0.4)
    assert not res.short_long.shown                  # confidence-gated off


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _miscalibrated_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    margins = rng.normal(0, 2.0, n)
    # True prob is sigmoid(margin / 3) — the raw sigmoid(margin) is over-confident.
    p_true = 1 / (1 + np.exp(-margins / 3.0))
    labels = (rng.random(n) < p_true).astype(float)
    return margins, labels


def test_temperature_scaling_reduces_ece():
    margins, labels = _miscalibrated_data()
    raw = 1 / (1 + np.exp(-margins))
    ece_raw = expected_calibration_error(raw, labels)
    scaler = TemperatureScaler().fit(margins, labels)
    ece_cal = expected_calibration_error(scaler.predict(margins), labels)
    assert ece_cal < ece_raw
    assert scaler.T > 1.0                            # recovers the over-confidence (T>1)


def test_platt_scaling_reduces_ece():
    margins, labels = _miscalibrated_data(seed=1)
    raw = 1 / (1 + np.exp(-margins))
    platt = PlattScaler().fit(margins, labels)
    assert expected_calibration_error(platt.predict(margins), labels) < expected_calibration_error(raw, labels)


def test_reliability_curve_and_brier():
    margins, labels = _miscalibrated_data(seed=2)
    scaler = TemperatureScaler().fit(margins, labels)
    probs = scaler.predict(margins)
    centers, acc, conf, count = reliability_curve(probs, labels, n_bins=10)
    ok = count > 0
    # After calibration, per-bin accuracy tracks confidence reasonably well.
    assert np.nanmean(np.abs(acc[ok] - conf[ok])) < 0.1
    assert 0.0 <= brier_score(probs, labels) <= 0.25
