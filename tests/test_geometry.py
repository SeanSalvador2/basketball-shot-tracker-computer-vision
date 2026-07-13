"""M2 geometry tests: projection round-trips, DLT correctness (+ cv2 cross-check),
the error-vs-noise curve matching the h/sin^2(phi) model, zones, and the rim ellipse."""
from __future__ import annotations

import numpy as np
import pytest

from bball.lift.court_model import (
    classify_with_band,
    classify_zone,
    classify_zone_radial,
    get_court,
    landmark_points,
)
from bball.lift.homography import (
    apply_homography,
    dlt_homography,
    estimate_homography,
    refine_homography_lm,
    reprojection_errors,
)
from bball.lift.projection import Camera
from bball.lift.rim_frame import RimEllipse, conic_to_geometric, fit_ellipse, rim_3d_center, rim_circle_3d


# --------------------------------------------------------------------------- #
# Projection primitives
# --------------------------------------------------------------------------- #
def test_ground_homography_matches_full_projection():
    """The ground-plane homography must agree with the full 3D pinhole projection for Z=0."""
    cam = Camera.from_look_at(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    court = get_court("nba")
    pts = np.array(list(landmark_points(court).values()))
    via_proj = cam.project(np.hstack([pts, np.zeros((pts.shape[0], 1))]))
    via_H = apply_homography(cam.ground_homography(), pts)
    assert np.allclose(via_proj, via_H, atol=1e-6)


def test_depression_angle_and_error_model_monotonic():
    """Predicted ground error must fall as the camera is elevated (larger depression
    angle) — the qualitative content of the h/sin^2(phi) model."""
    pt = (0.0, 7.0)  # a point out near the arc
    low = Camera.from_look_at(azimuth_deg=45, height_m=1.5, distance_m=9.0)
    high = Camera.from_look_at(azimuth_deg=45, height_m=3.5, distance_m=9.0)
    assert high.depression_angle(pt) > low.depression_angle(pt)
    assert high.predicted_ground_error(pt, sigma_px=3.0) < low.predicted_ground_error(pt, sigma_px=3.0)


# --------------------------------------------------------------------------- #
# DLT / homography
# --------------------------------------------------------------------------- #
def _court_correspondences(cam, court):
    pts_court = np.array(list(landmark_points(court).values()))
    pts_img = cam.project(np.hstack([pts_court, np.zeros((pts_court.shape[0], 1))]))
    return pts_court, pts_img


def test_dlt_recovers_homography_exactly_noise_free():
    cam = Camera.from_look_at(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    court = get_court("nba")
    src, dst = _court_correspondences(cam, court)
    H = dlt_homography(src, dst)
    err = reprojection_errors(H, src, dst)
    assert err.max() < 1e-6
    # And H matches the analytic ground homography up to scale.
    Hg = cam.ground_homography()
    ratio = H / Hg
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-4)


def test_dlt_cross_check_against_cv2():
    cv2 = pytest.importorskip("cv2")
    cam = Camera.from_look_at(azimuth_deg=60, height_m=2.5, distance_m=8.0)
    court = get_court("fiba")
    src, dst = _court_correspondences(cam, court)
    H_ours = dlt_homography(src, dst)
    H_cv, _ = cv2.findHomography(src.astype(np.float64), dst.astype(np.float64), 0)
    H_cv = H_cv / H_cv[2, 2]
    # Both map source points to the same image points.
    assert np.allclose(apply_homography(H_ours, src), apply_homography(H_cv, src), atol=1e-4)


def test_hartley_normalization_helps_conditioning():
    """Un-normalized DLT on raw pixel coordinates is ill-conditioned; our normalized DLT
    should stay accurate even at large pixel magnitudes."""
    cam = Camera.from_look_at(azimuth_deg=30, height_m=3.0, distance_m=10.0)
    court = get_court("nba")
    src, dst = _court_correspondences(cam, court)
    H = dlt_homography(src, dst)
    assert reprojection_errors(H, src, dst).max() < 1e-5


def test_lm_refinement_reduces_error_under_noise():
    rng = np.random.default_rng(0)
    cam = Camera.from_look_at(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    court = get_court("nba")
    src, dst = _court_correspondences(cam, court)
    noisy = dst + rng.normal(0, 2.0, size=dst.shape)
    H_dlt = dlt_homography(src, noisy)
    H_lm = refine_homography_lm(H_dlt, src, noisy)
    e_dlt = np.sqrt((reprojection_errors(H_dlt, src, noisy) ** 2).mean())
    e_lm = np.sqrt((reprojection_errors(H_lm, src, noisy) ** 2).mean())
    assert e_lm <= e_dlt + 1e-9  # LM never worse on the fitted residual


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    cam = Camera.from_look_at(azimuth_deg=45, height_m=3.0, distance_m=9.0)
    court = get_court("nba")
    src, dst = _court_correspondences(cam, court)
    # Duplicate the set to have enough points, then corrupt two as gross outliers.
    src2 = np.vstack([src, src + np.array([1.0, 0.0]), src - np.array([0.0, 1.0])])
    dst2 = apply_homography(cam.ground_homography(), src2)
    dst2[0] += np.array([120.0, -90.0])
    dst2[5] += np.array([-100.0, 80.0])
    res = estimate_homography(src2, dst2, use_ransac=True, refine=True, threshold_px=3.0, seed=3)
    assert not res.inlier_mask[0]
    assert not res.inlier_mask[5]
    assert res.n_inliers >= src2.shape[0] - 2


def test_error_grows_toward_grazing_view_monte_carlo():
    """Empirical A7 sanity check: propagate click noise to court-position error and
    confirm it is larger for a low grazing camera than a high one (matches h/sin^2 phi)."""
    court = get_court("nba")
    src = np.array(list(landmark_points(court).values()))
    probe = np.array([-6.0, 1.0])  # weak-side, near the corner-3 — a stressed location

    def median_court_error(height_m, n=120, seed=7):
        cam = Camera.from_look_at(azimuth_deg=25, height_m=height_m, distance_m=11.0)
        dst = cam.project(np.hstack([src, np.zeros((src.shape[0], 1))]))
        probe_img = cam.project(np.array([[probe[0], probe[1], 0.0]]))[0]
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(n):
            noisy = dst + rng.normal(0, 3.0, size=dst.shape)
            H = dlt_homography(src, noisy)  # court -> image
            Hinv = np.linalg.inv(H)
            est = apply_homography(Hinv, probe_img[None, :])[0]
            errs.append(np.hypot(*(est - probe)))
        return float(np.median(errs))

    e_low = median_court_error(1.5)
    e_high = median_court_error(3.5)
    assert e_high < e_low  # elevating the camera reduces court-position error


# --------------------------------------------------------------------------- #
# Court zones
# --------------------------------------------------------------------------- #
def test_zone_classification_key_positions():
    court = get_court("nba")
    assert classify_zone(court, 0.0, 1.0) == "short-range"     # in the paint
    assert classify_zone(court, 0.0, 5.0) == "midrange"        # elbow-ish, inside arc
    assert classify_zone(court, 0.0, 8.0) == "3PT"             # top of key beyond arc
    assert classify_zone(court, 6.9, 0.5) == "3PT"             # corner three
    assert classify_zone(court, 6.5, 0.5) == "midrange"        # just inside the corner line


def test_on_line_band_flags_boundary():
    court = get_court("nba")
    res = classify_with_band(court, 0.0, court.three_arc_radius_m, on_line_band_m=0.15)
    assert res["on_line"] is True
    res2 = classify_with_band(court, 0.0, court.three_arc_radius_m - 1.0, on_line_band_m=0.15)
    assert res2["on_line"] is False


def test_radial_mode_is_reasonable():
    court = get_court("nba")
    assert classify_zone_radial(court, 0.0, 1.0) == "short-range"
    assert classify_zone_radial(court, 0.0, 8.0) == "3PT"


# --------------------------------------------------------------------------- #
# Rim ellipse
# --------------------------------------------------------------------------- #
def test_rim_ellipse_roundtrip_from_projected_circle():
    """Project the 3D rim circle, fit an ellipse, and verify rim-normalized coordinates:
    the rim centre maps to ~0 and the projected rim points to radial fraction ~1."""
    cam = Camera.from_look_at(azimuth_deg=45, height_m=3.0, distance_m=9.0, target_height_m=0.0)
    circ3d = rim_circle_3d((0.0, 0.0), rim_height_m=3.048, rim_diameter_m=0.4572, n=80)
    circ_img = cam.project(circ3d)
    ell = RimEllipse.from_points(circ_img)
    center_img = cam.project(rim_3d_center((0.0, 0.0))[None, :])[0]
    frac_center = ell.radial_fraction(center_img[None, :])[0]
    frac_edge = ell.radial_fraction(circ_img)
    assert frac_center < 0.15
    assert np.allclose(frac_edge, 1.0, atol=0.05)


def test_ellipse_fit_recovers_known_ellipse():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    a, b, cx, cy, th = 40.0, 20.0, 300.0, 200.0, np.radians(25)
    x = cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
    y = cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
    center, sa, sb, theta = conic_to_geometric(fit_ellipse(np.stack([x, y], axis=1)))
    assert np.allclose(center, [cx, cy], atol=1e-3)
    assert abs(sa - a) < 1e-2 and abs(sb - b) < 1e-2


def test_rim_contains_and_margin_signs():
    ell = RimEllipse(cx=100.0, cy=100.0, a=30.0, b=15.0, theta_deg=0.0)
    assert ell.contains(np.array([[100.0, 100.0]]))[0]        # centre inside
    assert not ell.contains(np.array([[100.0, 130.0]]))[0]    # outside the minor axis
    assert ell.interior_margin(np.array([[100.0, 100.0]]))[0] > 0
    assert ell.interior_margin(np.array([[131.0, 100.0]]))[0] < 0  # just past major axis
