"""Zone-partition tests: coverage, corner-offset correctness, bands, lifting, reliability."""
import numpy as np
import pytest

from bball.lift.court_model import get_court
from bball.lift.homography import apply_homography
from bball.lift.zones import (
    ZonePartition,
    compose_with_error_map,
    distance_beyond_three,
    from_dict,
    from_polygons,
    lift_screen_polyline,
    offset_three_polyline,
    preset_basic3,
    preset_extended,
    preset_spots,
)

NBA = get_court("nba")


def _half_court_grid(n=40):
    c = NBA
    xs = np.linspace(-c.sideline_x_m + 0.05, c.sideline_x_m - 0.05, n)
    ys = np.linspace(-c.rim_from_baseline_m + 0.05, 11.0, n)
    return [(float(x), float(y)) for x in xs for y in ys]


@pytest.mark.parametrize("part_fn", [preset_basic3, preset_extended, preset_spots])
def test_presets_cover_half_court_with_single_labels(part_fn):
    part = part_fn(NBA)
    for x, y in _half_court_grid():
        z = part.label(x, y)
        assert z in part.zones, f"unknown zone {z} at {(x, y)}"


def test_basic3_matches_taxonomy():
    part = preset_basic3(NBA)  # 7 ft interior arc
    assert part.label(0.0, 1.0) == "interior"
    assert part.label(0.0, 4.0) == "midrange"          # beyond 7 ft, inside arc
    assert part.label(0.0, 7.5) == "three"             # beyond 7.239 m apex
    assert part.label(7.0, 0.5) == "three"             # corner three (radius < apex!)


def test_corner_three_is_not_radial():
    # hypot(6.9, 0.5) = 6.92 < arc radius 7.239 — a radial rule would say midrange.
    assert np.hypot(6.9, 0.5) < NBA.three_arc_radius_m
    assert distance_beyond_three(NBA, 6.9, 0.5) > 0
    assert preset_basic3(NBA).label(6.9, 0.5) == "three"


def test_deep_three_uses_offset_of_true_shape_not_radius():
    part = preset_extended(NBA, deep_three_offset_m=0.9)
    # Corner point beyond xc + 0.9: offset logic says deep-three...
    x, y = 7.8, 0.3
    assert part.label(x, y) == "deep-three"
    # ...but the naive radial test (r > R_arc + 0.9) would NOT have fired.
    assert np.hypot(x, y) < NBA.three_arc_radius_m + 0.9
    # Above the transition, the arc rules: apex + 0.76 m is three, not deep-three.
    assert part.label(0.0, NBA.three_arc_radius_m + 0.76) == "three"
    assert part.label(0.0, NBA.three_arc_radius_m + 0.95) == "deep-three"


def test_offset_polyline_transition_recomputed():
    d = 0.9
    poly = offset_three_polyline(NBA, d)
    R, xc = NBA.three_arc_radius_m + d, NBA.corner_three_dist_m + d
    yt = np.sqrt(R * R - xc * xc)
    # Corner verticals sit at ±xc and reach the recomputed transition height.
    assert np.isclose(np.max(poly[:, 0]), xc, atol=1e-6)
    assert np.isclose(np.max(poly[np.isclose(poly[:, 0], xc), 1]), yt, atol=1e-6)


def test_on_line_band_flags_near_boundary():
    part = preset_basic3(NBA)
    near = part.classify(0.0, NBA.three_arc_radius_m - 0.05, band_m=0.15)
    far = part.classify(0.0, 4.5, band_m=0.15)
    assert near["on_line"] and near["nearest_boundary"] == "three_line"
    assert not far["on_line"]


def test_spots_sectors():
    part = preset_spots(NBA)
    assert part.label(7.0, 0.4) == "right-corner-three"
    assert part.label(-7.0, 0.4) == "left-corner-three"
    assert part.label(0.0, 8.0) == "top-three"
    assert part.label(0.0, 5.0) == "top-mid"
    assert part.label(0.0, 1.0) == "interior"
    lab = part.label(4.4, 3.6)  # ~39 deg off baseline, r=5.7: wing midrange
    assert lab == "right-wing-mid"


def test_freeform_polygons_and_default():
    sq = [(0, 2), (2, 2), (2, 4), (0, 4)]
    part = from_polygons("custom", {"my-spot": sq}, default_zone="elsewhere")
    assert part.label(1.0, 3.0) == "my-spot"
    assert part.label(5.0, 5.0) == "elsewhere"
    assert "my-spot_outline" in part.boundaries


def test_serialization_round_trip():
    for part in (preset_basic3(NBA), preset_extended(NBA), preset_spots(NBA),
                 from_polygons("c", {"z": [(0, 0), (1, 0), (1, 1)]})):
        clone = from_dict(part.to_dict())
        for x, y in [(0.0, 1.0), (5.0, 2.0), (7.8, 0.3), (0.0, 8.5)]:
            assert clone.label(x, y) == part.label(x, y)


def test_rebucket_is_pure_relabel():
    part_a, part_b = preset_basic3(NBA), preset_extended(NBA)
    shots = [{"court_xy": (0.0, 8.5), "outcome": "make"},
             {"court_xy": (0.0, 4.0), "outcome": "miss"}]
    ra, rb = part_a.rebucket(shots), part_b.rebucket(shots)
    assert [r["zone"] for r in ra] == ["three", "midrange"]
    assert [r["zone"] for r in rb] == ["deep-three", "short-mid"]  # 8.5 m is 1.26 m beyond the arc
    assert all(r["partition"] == "basic3" for r in ra)
    assert shots[0].get("zone") is None  # inputs untouched


def test_lift_screen_polyline_round_trip_and_validation():
    H = np.array([[1.2, 0.1, 50.0], [-0.05, 1.1, 200.0], [2e-4, 5e-4, 1.0]])
    court_pts = np.array([[0.0, 0.0], [3.0, 4.0], [-6.0, 1.0], [0.0, 7.24]])
    screen = apply_homography(H, court_pts)
    lifted = lift_screen_polyline(H, screen, court=NBA)
    assert np.allclose(lifted, court_pts, atol=1e-6)
    with pytest.raises(ValueError):
        lift_screen_polyline(np.eye(3), np.array([[5000.0, 5000.0], [5100.0, 5000.0]]), court=NBA)


def test_error_map_composition_verdicts():
    part = preset_basic3(NBA)
    good = compose_with_error_map(part, lambda x, y: 0.05, band_m=0.15)
    bad = compose_with_error_map(part, lambda x, y: 0.50, band_m=0.15)
    assert all(v["verdict"] == "ok" and v["reliable_fraction"] == 1.0 for v in good.values())
    assert all(v["verdict"] == "unreliable" for v in bad.values())
