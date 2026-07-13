"""M7 eval-harness tests: metrics, the anti-leakage split guard, bootstrap CIs, and a
figure smoke test."""
from __future__ import annotations

import numpy as np
import pytest

from bball.eval.bootstrap import bootstrap_ci, paired_session_delta
from bball.eval.metrics import (
    event_prf,
    false_attempts_per_hour,
    outcome_prf,
    per_axis_t5_accuracy,
    t3_error_cm,
    track_completeness,
    zone_confusion,
)
from bball.eval.splits import assert_no_leakage, assert_test_venue_held_out, make_split


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_event_prf_tolerance_matching():
    gt = [1.0, 2.0, 5.0]
    pred = [1.1, 2.05, 9.0]           # 2 within 0.25s, 1 false, 1 missed
    prf = event_prf(pred, gt, tol_s=0.25)
    assert prf.tp == 2 and prf.fp == 1 and prf.fn == 1
    assert prf.precision == pytest.approx(2 / 3)
    assert prf.recall == pytest.approx(2 / 3)


def test_event_prf_perfect():
    prf = event_prf([1.0, 2.0], [1.0, 2.0])
    assert prf.f1 == pytest.approx(1.0)


def test_outcome_prf():
    prf = outcome_prf(["make", "miss", "make", "make"], ["make", "miss", "miss", "make"])
    assert prf.tp == 2 and prf.fp == 1 and prf.fn == 0


def test_false_attempts_per_hour():
    assert false_attempts_per_hour(3, 1800) == pytest.approx(6.0)  # 3 in half an hour


def test_t3_error_cm():
    pred = np.array([[0.0, 0.0], [1.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0]])
    res = t3_error_cm(pred, gt)
    assert res["median_cm"] == pytest.approx(5.0, abs=0.1)


def test_zone_confusion_accuracy():
    res = zone_confusion(["3PT", "midrange", "3PT"], ["3PT", "midrange", "midrange"])
    assert res["accuracy"] == pytest.approx(2 / 3)


def test_track_completeness_split_by_occlusion():
    observed = np.array([1, 1, 0, 0, 1, 1], bool)
    occ = np.array([0.0, 0.0, 0.9, 0.9, 0.1, 0.1])
    res = track_completeness(observed, occlusion=occ)
    assert res["observed_pct"] == pytest.approx(4 / 6)
    assert res["observed_pct_clean"] == pytest.approx(1.0)
    assert res["observed_pct_occluded"] == pytest.approx(0.0)


def test_per_axis_t5_reported_separately():
    preds = [{"left_right": "left", "short_long": "long"}, {"left_right": "right", "short_long": "short"}]
    gts = [{"left_right": "left", "short_long": "short"}, {"left_right": "right", "short_long": "short"}]
    res = per_axis_t5_accuracy(preds, gts)
    assert res["left_right"]["accuracy"] == pytest.approx(1.0)   # both correct
    assert res["short_long"]["accuracy"] == pytest.approx(0.5)   # one correct


# --------------------------------------------------------------------------- #
# Splits / leakage
# --------------------------------------------------------------------------- #
def _scenes():
    scenes = []
    for venue in ["gym_A", "gym_B", "outdoor_A", "outdoor_B"]:
        for k in range(4):
            scenes.append({"scene_id": f"{venue}_{k}", "venue": venue})
    return scenes


def test_split_holds_out_test_venue_no_leakage():
    scenes = _scenes()
    split = make_split(scenes, test_venues=["outdoor_B"], val_frac=0.3, seed=0)
    assert_no_leakage(split)
    assert_test_venue_held_out(split, scenes, ["outdoor_B"])
    assert len(split.test) == 4                       # all outdoor_B scenes
    assert not (set(split.val_tune) & set(split.val_cal))   # val-tune/val-cal disjoint (R6)


def test_leakage_guard_catches_overlap():
    from bball.eval.splits import Split

    bad = Split(train=["a", "b"], val_tune=["b"], val_cal=[], test=["c"])
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_val_tune_and_cal_partition_val():
    scenes = _scenes()
    split = make_split(scenes, test_venues=["gym_A"], val_frac=0.5, val_cal_frac=0.5, seed=1)
    assert sorted(split.val) == sorted(split.val_tune + split.val_cal)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_point():
    vals = [0.9, 0.92, 0.88, 0.91, 0.87, 0.93]
    ci = bootstrap_ci(vals, n_boot=1000, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_paired_delta_detects_improvement():
    a = [0.95, 0.96, 0.94, 0.97, 0.95]
    b = [0.90, 0.91, 0.89, 0.92, 0.90]
    res = paired_session_delta(a, b, n_boot=1000, seed=0)
    assert res["delta"] > 0 and res["significant"]


# --------------------------------------------------------------------------- #
# Viz smoke
# --------------------------------------------------------------------------- #
def test_viz_and_gallery_smoke(tmp_path):
    from bball.eval.galleries import failure_contact_sheet
    from bball.lift.rim_frame import RimEllipse
    from bball.viz import reliability_diagram, shot_chart
    import matplotlib.pyplot as plt

    ax = shot_chart([{"xy": (0, 5), "outcome": "make"}, {"xy": (6, 1), "outcome": "miss"}])
    assert ax is not None
    plt.close("all")

    rng = np.random.default_rng(0)
    margins = rng.normal(0, 2, 500)
    labels = (rng.random(500) < 1 / (1 + np.exp(-margins))).astype(float)
    reliability_diagram({"raw": 1 / (1 + np.exp(-margins))}, labels)
    plt.close("all")

    ell = RimEllipse(cx=100, cy=100, a=30, b=8, theta_deg=0)
    traj = np.stack([np.linspace(70, 130, 20), np.linspace(60, 140, 20)], axis=1)
    out = failure_contact_sheet([{"ball_img": traj, "rim_ellipse": ell, "caption": "test"}],
                                tmp_path / "gallery.png")
    assert out.exists()
