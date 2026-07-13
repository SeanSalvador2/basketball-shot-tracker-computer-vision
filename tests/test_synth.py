"""M3 synthetic-engine tests: physics labelling, projection + ball size, occlusion-driven
detection gaps (the input A5 bridges), rendering + mp4 writing, session generation."""
from __future__ import annotations

import numpy as np
import pytest

from bball.synth.camera import apparent_ball_radius_px, make_camera, project_trajectory
from bball.synth.physics import (
    BALL_DIAMETER_M,
    RIM_HEIGHT_M,
    RIM_RADIUS_M,
    generate_shot,
    solve_launch,
)
from bball.synth.render import (
    DetectionNoiseModel,
    SceneAppearance,
    compute_rim_image_geometry,
    render_clip,
    write_mp4,
)
from bball.synth.scenarios import default_shot_script, generate_session, simulate_players, venue_scene
from bball.lift.court_model import get_court


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def test_make_passes_through_rim_and_clears_it():
    s = generate_shot(release_xy=(0.0, 6.5), outcome="make", release_angle_deg=50, seed=1)
    assert s.apex_height_m > RIM_HEIGHT_M
    i = int(np.argmin(np.abs(s.t - s.events["rim_arrival_t"])))
    assert np.hypot(*s.pos[i, :2]) < RIM_RADIUS_M      # inside the rim at arrival


def test_short_miss_is_airball_outside_rim():
    s = generate_shot(release_xy=(0.0, 6.5), outcome="miss", miss_direction="short",
                      miss_magnitude_m=1.2, seed=2)
    i = int(np.argmin(np.abs(s.t - s.events["rim_arrival_t"])))
    assert np.hypot(*s.pos[i, :2]) > RIM_RADIUS_M


def test_lateral_miss_offsets_correct_side():
    right = generate_shot(release_xy=(0.0, 6.5), outcome="miss", miss_direction="right",
                          miss_magnitude_m=0.6, seed=3)
    left = generate_shot(release_xy=(0.0, 6.5), outcome="miss", miss_direction="left",
                         miss_magnitude_m=0.6, seed=3)
    ir = int(np.argmin(np.abs(right.t - right.events["rim_arrival_t"])))
    il = int(np.argmin(np.abs(left.t - left.events["rim_arrival_t"])))
    assert right.pos[ir, 0] > 0 and left.pos[il, 0] < 0   # +x is the shooter's right here


def test_solve_launch_reaches_target():
    v, beta, tof = solve_launch(np.array([0, 6.5, 2.2]), np.array([0, 0]), RIM_HEIGHT_M, 50.0)
    assert v > 0 and tof > 0


def test_release_before_apex_before_rim():
    s = generate_shot(release_xy=(3.0, 6.0), outcome="make", seed=5)
    e = s.events
    assert e["release_t"] < e["apex_t"] < e["rim_arrival_t"]


# --------------------------------------------------------------------------- #
# Camera / projection
# --------------------------------------------------------------------------- #
def test_ball_radius_shrinks_with_distance():
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    near = np.array([[0.0, 1.0, 2.0]])
    far = np.array([[0.0, 8.0, 2.0]])
    assert apparent_ball_radius_px(cam, near)[0] > apparent_ball_radius_px(cam, far)[0]


def test_ball_radius_in_expected_regime():
    """Phase-0 assumed a 20-40 px ball regime; a mid-court shot at 1080p should land in a
    plausible small-object range (single-digit to tens of px radius)."""
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    r = apparent_ball_radius_px(cam, np.array([[0.0, 5.0, 2.2]]))[0]
    assert 2.0 < r < 60.0


# --------------------------------------------------------------------------- #
# Occlusion + detection noise (the A5 input)
# --------------------------------------------------------------------------- #
def test_occlusion_creates_detection_gap_near_rim():
    """A make passing through the rim/net must produce missed detections there — the gap
    the bridging layer (A5) has to close."""
    scene = venue_scene("gym_A", azimuth_deg=45, height_m=3.0)
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
    shot = generate_shot(release_xy=(0.0, 6.5), outcome="make", rattle=False, seed=7)
    ball_img = project_trajectory(cam, shot.pos)
    ball_rad = apparent_ball_radius_px(cam, shot.pos)
    from bball.synth.render import occlusion_fraction
    occl = np.array([occlusion_fraction(None if np.isnan(ball_img[i]).any() else ball_img[i],
                                        float(ball_rad[i]) if not np.isnan(ball_rad[i]) else 3.0, rim_geom)
                     for i in range(shot.n_frames)])
    assert occl.max() > 0.5                       # ball is occluded somewhere near the rim
    noise = DetectionNoiseModel()
    stream = noise.stream(ball_img, ball_rad, occl, np.random.default_rng(0))
    n_missed = sum(1 for c in stream if not c.observed)
    assert n_missed >= 1                           # at least one occlusion-driven miss
    # And clean-flight frames are mostly observed.
    clean_obs = sum(1 for i, c in enumerate(stream) if occl[i] < 0.1 and c.observed)
    clean_total = sum(1 for i in range(len(stream)) if occl[i] < 0.1)
    assert clean_obs / max(clean_total, 1) > 0.8


def test_detection_jitter_bounded_on_clean_flight():
    noise = DetectionNoiseModel(base_miss_prob=0.0, jitter_px=1.5)
    rng = np.random.default_rng(1)
    gt = np.array([500.0, 300.0])
    errs = []
    for _ in range(500):
        c = noise.sample(0, gt, 10.0, occlusion=0.0, blur_len_px=0.0, rng=rng)
        errs.append(np.hypot(*(c.xy - gt)))
    assert np.median(errs) < 3.0                   # clean localization is tight


# --------------------------------------------------------------------------- #
# Rendering + mp4
# --------------------------------------------------------------------------- #
def test_render_and_write_mp4(tmp_path):
    scene = venue_scene("gym_A", azimuth_deg=45, height_m=3.0)
    cam = make_camera(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    shot = generate_shot(release_xy=(2.0, 6.0), outcome="make", fps=30, pre_release_s=0.2,
                         post_rim_s=0.2, seed=8)
    out = render_clip(shot, cam, scene.court, scene.appearance, scale=0.33)
    frames = out["frames"]
    assert len(frames) == shot.n_frames
    assert frames[0].shape[2] == 3 and frames[0].dtype == np.uint8
    path = tmp_path / "clip.mp4"
    write_mp4(frames, str(path), fps=30)
    assert path.exists() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Session generation
# --------------------------------------------------------------------------- #
def test_generate_session_labels_and_negatives():
    scene = venue_scene("gym_B", azimuth_deg=30, height_m=1.5)
    sess = generate_session(scene, n_shots=20, fps=60, seed=3, n_negatives=3)
    assert len(sess.shots) == 20
    assert len(sess.negatives) == 3
    outcomes = {s.outcome for s in sess.shots}
    assert outcomes <= {"make", "miss"}
    assert any(s.outcome == "make" for s in sess.shots)
    assert any(s.outcome == "miss" for s in sess.shots)
    assert all(n.meta.get("is_negative") for n in sess.negatives)
    rows = sess.events_table()
    assert len(rows) == 20 and "rim_arrival_t" in rows[0]


def test_players_walk_within_reason():
    court = get_court("nba")
    players = simulate_players(3, 120, court, fps=60, seed=0)
    assert len(players) == 3
    for p in players:
        assert p["pos_xy"].shape == (120, 2)
        step = np.linalg.norm(np.diff(p["pos_xy"], axis=0), axis=1)
        assert step.max() < 0.2       # <= ~ walking speed / fps
