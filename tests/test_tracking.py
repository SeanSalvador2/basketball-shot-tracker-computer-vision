"""M5 tracking tests: Kalman vs analytic cases, association + ByteTrack recovery,
Level-1 bridging (gaps + multi-ball rejection), Level-2 3D reconstruction + azimuth
observability (the A6 mechanism)."""
from __future__ import annotations

import numpy as np
import pytest

from bball.detect.interfaces import BallCandidate, Detection
from bball.lift.rim_frame import rim_3d_center
from bball.synth.camera import make_camera, project_trajectory, apparent_ball_radius_px
from bball.synth.physics import generate_shot
from bball.synth.scenarios import simulate_players
from bball.lift.court_model import get_court
from bball.track.association import ByteTrackPlayerTracker, associate_iou, gated_nearest, iou_matrix
from bball.track.ballistic import (
    bridge_trajectory,
    fit_level1,
    miss_vector_rim_local,
    reconstruct_flight_3d,
)
from bball.track.kalman import KalmanFilter, cv_kalman_2d


# --------------------------------------------------------------------------- #
# Kalman
# --------------------------------------------------------------------------- #
def test_kalman_predict_coasts_at_constant_velocity():
    kf = cv_kalman_2d((0.0, 0.0), dt=1.0, init_vel=(2.0, -1.0))
    kf.predict()
    assert np.allclose(kf.x[:2], [2.0, -1.0])
    kf.predict()
    assert np.allclose(kf.x[:2], [4.0, -2.0])


def test_kalman_denoises_constant_velocity():
    """The fundamental KF property: with a matched model, the filtered estimate has lower
    error than the raw measurements, and velocity is recovered."""
    rng = np.random.default_rng(0)
    kf = cv_kalman_2d((0.0, 0.0), dt=1.0, process_std=0.1, meas_std=2.0, init_vel=(0.0, 0.0))
    true = np.array([0.0, 0.0]); vel = np.array([3.0, 1.5])
    filt_err, meas_err = [], []
    for k in range(60):
        true = true + vel
        z = true + rng.normal(0, 2.0, 2)
        kf.predict(); kf.update(z)
        filt_err.append(np.hypot(*(kf.x[:2] - true)))
        meas_err.append(np.hypot(*(z - true)))
    assert np.mean(filt_err[10:]) < np.mean(meas_err[10:])   # denoises the measurements
    assert np.allclose(kf.x[2:4], vel, atol=0.4)             # recovers velocity


def test_kalman_variance_decreases_on_update():
    kf = cv_kalman_2d((0.0, 0.0))
    p_before = np.trace(kf.P)
    kf.predict(); kf.update((0.1, -0.1))
    assert np.trace(kf.P) < p_before


def test_mahalanobis_gating_orders_by_distance():
    kf = cv_kalman_2d((0.0, 0.0), meas_std=2.0)
    kf.predict()
    near = kf.mahalanobis2((0.0, 0.0))
    far = kf.mahalanobis2((100.0, 100.0))
    assert far > near


def test_general_kf_matches_scalar_analytic():
    # 1D position-only random walk: steady-state gain is known-positive; filter reduces MSE.
    F = np.array([[1.0]]); H = np.array([[1.0]]); Q = np.array([[1e-4]]); R = np.array([[1.0]])
    kf = KalmanFilter(F, H, Q, R, [0.0], [[1.0]])
    rng = np.random.default_rng(1)
    true = 5.0
    est = []
    for _ in range(50):
        kf.predict(); kf.update([true + rng.normal(0, 1.0)])
        est.append(kf.x[0])
    assert abs(np.mean(est[-10:]) - true) < 0.6


# --------------------------------------------------------------------------- #
# Association / ByteTrack
# --------------------------------------------------------------------------- #
def test_associate_iou_matches_overlap():
    tb = [(0, 0, 10, 10), (100, 100, 110, 110)]
    db = [(101, 101, 111, 111), (1, 1, 11, 11)]
    matches, ut, ud = associate_iou(tb, db, iou_threshold=0.2)
    assert (0, 1) in matches and (1, 0) in matches
    assert not ut and not ud


def _player_box(camera, xy, height_m):
    w = 0.55
    corners = np.array([[xy[0] - w / 2, xy[1], 0.0], [xy[0] + w / 2, xy[1], 0.0], [xy[0], xy[1], height_m]])
    p = camera.project(corners)
    x0, x1 = min(p[0, 0], p[1, 0]), max(p[0, 0], p[1, 0])
    ytop, ybot = p[2, 1], max(p[0, 1], p[1, 1])
    return (float(x0), float(ytop), float(x1), float(ybot))


def test_bytetrack_keeps_ids_stable_on_two_players():
    court = get_court("nba")
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    players = simulate_players(2, 60, court, fps=30, seed=1)
    tracker = ByteTrackPlayerTracker()
    id_sets = []
    for i in range(60):
        dets = [Detection(bbox=_player_box(cam, p["pos_xy"][i], p["height_m"]), score=0.9,
                          label="person", frame_idx=i) for p in players]
        out = tracker.update(dets)
        id_sets.append(set(out.keys()))
    # After warmup, exactly two stable tracks and few distinct ids overall.
    steady = id_sets[10:]
    assert all(len(s) == 2 for s in steady[-20:])
    all_ids = set().union(*id_sets)
    assert len(all_ids) <= 4         # allow a little churn, but not a new id per frame


def test_bytetrack_low_score_recovery_survives_weak_frame():
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    tracker = ByteTrackPlayerTracker()
    box = (500.0, 300.0, 540.0, 420.0)
    for i in range(5):
        tracker.update([Detection(bbox=box, score=0.9, label="person", frame_idx=i)])
    # a weak (low-score) detection should still keep the track alive, not spawn a new id
    ids_before = set(tracker.update([Detection(bbox=box, score=0.2, label="person", frame_idx=5)]).keys())
    ids_after = set(tracker.update([Detection(bbox=box, score=0.9, label="person", frame_idx=6)]).keys())
    assert ids_before == ids_after
    assert len(ids_after) == 1


def test_gated_nearest_picks_closest_in_gate():
    cands = [BallCandidate(0, np.array([10.0, 10.0]), 0.5), BallCandidate(0, np.array([100.0, 100.0]), 0.9)]
    best = gated_nearest(np.array([12.0, 9.0]), cands, gate_px=20.0)
    assert best is cands[0]
    assert gated_nearest(np.array([500.0, 500.0]), cands, gate_px=20.0) is None


# --------------------------------------------------------------------------- #
# Level-1 bridging
# --------------------------------------------------------------------------- #
def test_level1_fit_recovers_quadratic():
    t = np.linspace(0, 1, 20)
    xy = np.stack([3 * t + 1, -4.9 * t * t + 5 * t + 2], axis=1)
    fit = fit_level1(t, xy)
    pred = fit.predict(t)
    assert np.allclose(pred, xy, atol=1e-6)


def _stream_from_shot(cam, shot, drop_range=None):
    img = project_trajectory(cam, shot.pos)
    rad = apparent_ball_radius_px(cam, shot.pos)
    cands = []
    for i in range(shot.n_frames):
        drop = drop_range is not None and drop_range[0] <= i < drop_range[1]
        if np.isnan(img[i]).any() or drop:
            cands.append(BallCandidate(i, None))
        else:
            cands.append(BallCandidate(i, img[i], 0.9, float(rad[i]) if not np.isnan(rad[i]) else 3.0))
    return cands, img


def test_bridging_fills_gap_l1_but_not_off():
    cam = make_camera(azimuth_deg=90, height_m=3.0, distance_m=10.0)
    shot = generate_shot(release_xy=(-2.0, 6.0), outcome="miss", miss_direction="long", seed=6)
    cands, img = _stream_from_shot(cam, shot, drop_range=(20, 28))   # 8-frame occlusion
    t = shot.t
    off = bridge_trajectory(cands, t, method="off")
    l1 = bridge_trajectory(cands, t, method="l1")
    assert l1.completeness > off.completeness
    # bridged points in the gap should be near the true image trajectory
    gap = range(20, 28)
    errs = [np.hypot(*(l1.xy[i] - img[i])) for i in gap if not np.isnan(l1.xy[i]).any() and not np.isnan(img[i]).any()]
    assert len(errs) >= 5 and np.median(errs) < 25


def test_bridging_rejects_second_ball_outlier():
    cam = make_camera(azimuth_deg=90, height_m=3.0, distance_m=10.0)
    shot = generate_shot(release_xy=(0.0, 6.0), outcome="make", seed=7)
    cands, img = _stream_from_shot(cam, shot)
    # inject a physics-violating candidate (a second ball far away) mid-flight
    k = 22
    cands[k] = BallCandidate(k, img[k] + np.array([300.0, 250.0]), 0.9, 5.0)
    res = bridge_trajectory(cands, shot.t, method="l1", base_gate_px=40, gate_growth_px=10)
    assert not res.observed[k]            # the outlier was gated out
    assert res.bridged[k]                 # and replaced by the fit prediction


# --------------------------------------------------------------------------- #
# Level-2 reconstruction + azimuth observability (A6 mechanism)
# --------------------------------------------------------------------------- #
def test_l2_reconstructs_side_on_shot_accurately():
    cam = make_camera(azimuth_deg=90, height_m=3.0, distance_m=10.0)  # side-on
    shot = generate_shot(release_xy=(0.0, 6.5), outcome="miss", miss_direction="left",
                         miss_magnitude_m=0.5, seed=8)
    # use only the flight portion (release..rim arrival)
    mask = (shot.t >= shot.events["release_t"]) & (shot.t <= shot.events["rim_arrival_t"] + 0.05)
    pos = shot.pos[mask]; t = shot.t[mask]
    img = cam.project(pos); rad = apparent_ball_radius_px(cam, pos)
    traj = reconstruct_flight_3d(img, t, cam, shooter_feet_xy=shot.release_xy,
                                 rim_center_3d=rim_3d_center((0.0, 0.0)), ball_radius_px=rad)
    assert traj.rms_reproj_px < 5.0
    assert traj.confidence > 0.3
    mv = miss_vector_rim_local(traj, rim_3d_center((0.0, 0.0)), shot.release_xy)
    assert mv["left_right"] == "left"          # recovers the correct lateral side


def test_depth_observability_decreases_toward_end_on():
    """The A6 mechanism: a camera aligned with the shooting lane loses the depth axis."""
    shot_feet = np.array([0.0, 7.0])
    rim = rim_3d_center((0.0, 0.0))
    from bball.track.ballistic import _depth_observability
    side = _depth_observability(make_camera(azimuth_deg=90, height_m=3, distance_m=10), shot_feet, rim)
    endon = _depth_observability(make_camera(azimuth_deg=5, height_m=3, distance_m=10), shot_feet, rim)
    assert side > endon                       # depth better observed side-on
