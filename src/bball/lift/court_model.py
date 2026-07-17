"""Court geometry: dimensions, shot zones, on-the-line band, landmark-sparse radial mode.

Coordinate frame (metres): origin at the **hoop's ground projection** (point on the floor
under the rim centre); +Y points into the court (toward half-court), +X to the right.
This hoop-centred frame makes zones a function of radial distance, which is also the
degraded ("radial") fallback for courts without painted lines (review R10).

Every dimension is a published court spec (plan gate G4 — no fantasy constants):
NBA/FIBA/NFHS rulebook measurements, cited per field below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    name: str
    court_width_m: float          # sideline-to-sideline
    rim_from_baseline_m: float    # hoop centre to baseline
    three_arc_radius_m: float     # 3pt arc radius from hoop centre
    corner_three_dist_m: float    # straight corner-3 distance from hoop centre
    corner_from_sideline_m: float # corner-3 line offset from the sideline
    lane_half_width_m: float      # half the painted-lane width
    ft_line_from_hoop_m: float    # free-throw line distance from hoop centre (into court)
    restricted_radius_m: float    # restricted-area arc radius
    backboard_from_baseline_m: float

    @property
    def sideline_x_m(self) -> float:
        return self.court_width_m / 2.0

    @property
    def corner_transition_y_m(self) -> float:
        """Y at which the straight corner-3 line meets the arc (y_t = sqrt(R3^2 - xc^2))."""
        xc = self.corner_three_dist_m
        R = self.three_arc_radius_m
        return float(np.sqrt(max(R * R - xc * xc, 0.0)))


# Published specs. NBA: 50 ft width, rim 1.60 m from baseline, 23'9" arc / 22' corner,
# 16 ft lane, FT line 15 ft from backboard. FIBA: 15 m, 6.75/6.60 m, 4.9 m lane.
# NFHS (US high school): 50 ft, 19'9" uniform arc, 12 ft lane.
_SPECS: dict[str, CourtSpec] = {
    "nba": CourtSpec(
        name="nba", court_width_m=15.24, rim_from_baseline_m=1.6002,
        three_arc_radius_m=7.239, corner_three_dist_m=6.7056, corner_from_sideline_m=0.9144,
        lane_half_width_m=2.4384, ft_line_from_hoop_m=4.191, restricted_radius_m=1.2192,
        backboard_from_baseline_m=1.2192,
    ),
    "fiba": CourtSpec(
        name="fiba", court_width_m=15.0, rim_from_baseline_m=1.575,
        three_arc_radius_m=6.75, corner_three_dist_m=6.60, corner_from_sideline_m=0.90,
        lane_half_width_m=2.45, ft_line_from_hoop_m=4.225, restricted_radius_m=1.25,
        backboard_from_baseline_m=1.20,
    ),
    "hs": CourtSpec(
        name="hs", court_width_m=15.24, rim_from_baseline_m=1.575,
        three_arc_radius_m=6.02, corner_three_dist_m=6.02, corner_from_sideline_m=0.9144,
        lane_half_width_m=1.8288, ft_line_from_hoop_m=4.191, restricted_radius_m=1.2192,
        backboard_from_baseline_m=1.2192,
    ),
}

ZONES = ("short-range", "midrange", "3PT")


def get_court(spec: str | CourtSpec = "nba", **overrides) -> CourtSpec:
    """Fetch a named spec or build a custom one. `spec='custom'` requires all fields as
    keyword overrides; a named spec with overrides patches individual dimensions."""
    if isinstance(spec, CourtSpec):
        base = spec
    elif spec == "custom":
        return CourtSpec(name="custom", **overrides)
    else:
        key = spec.lower()
        if key not in _SPECS:
            raise ValueError(f"unknown court spec {spec!r}; options: {list(_SPECS)} or 'custom'")
        base = _SPECS[key]
    if overrides:
        return CourtSpec(**{**base.__dict__, **overrides})
    return base


# --------------------------------------------------------------------------- #
# Distances to boundaries (for the on-the-line band and viz)
# --------------------------------------------------------------------------- #
def distance_to_three_line(court: CourtSpec, x: float, y: float) -> float:
    """Unsigned distance (m) from a point to the 3-point boundary (arc + corner lines)."""
    r = np.hypot(x, y)
    yt = court.corner_transition_y_m
    d_arc = abs(r - court.three_arc_radius_m)
    d_corner = abs(abs(x) - court.corner_three_dist_m)
    if y <= yt:
        return float(d_corner)
    return float(d_arc)


def _in_paint(court: CourtSpec, x: float, y: float, radius_m: float) -> bool:
    in_lane = (abs(x) <= court.lane_half_width_m) and (0.0 <= y <= court.ft_line_from_hoop_m)
    near_hoop = np.hypot(x, y) <= radius_m
    return bool(in_lane or near_hoop)


def _is_three(court: CourtSpec, x: float, y: float) -> bool:
    yt = court.corner_transition_y_m
    if y <= yt:
        return abs(x) > court.corner_three_dist_m
    return np.hypot(x, y) > court.three_arc_radius_m


def classify_zone(
    court: CourtSpec, x: float, y: float, *, short_range_radius_m: float = 2.4
) -> str:
    """Analytic zone of a court point (hoop-centred metres)."""
    if _is_three(court, x, y):
        return "3PT"
    if _in_paint(court, x, y, short_range_radius_m):
        return "short-range"
    return "midrange"


def classify_with_band(
    court: CourtSpec,
    x: float,
    y: float,
    *,
    on_line_band_m: float = 0.15,
    short_range_radius_m: float = 2.4,
) -> dict:
    """Zone + on-the-line flag. The band is measured to the 3-point line (the boundary
    whose misclassification actually changes a shot's value)."""
    zone = classify_zone(court, x, y, short_range_radius_m=short_range_radius_m)
    d3 = distance_to_three_line(court, x, y)
    on_line = d3 <= on_line_band_m
    return {"zone": zone, "on_line": bool(on_line), "dist_to_3pt_m": d3}


# --------------------------------------------------------------------------- #
# Landmark-sparse radial mode (review R10)
# --------------------------------------------------------------------------- #
def classify_zone_radial(court: CourtSpec, x: float, y: float, *, short_range_radius_m: float = 2.4) -> str:
    """Degraded zoning from radial distance only (driveway courts, 4 tape markers).
    Bands: short-range < r1, midrange r1..R3, 3PT >= R3 (arc radius)."""
    r = np.hypot(x, y)
    if r >= court.three_arc_radius_m:
        return "3PT"
    if r <= short_range_radius_m:
        return "short-range"
    return "midrange"


# --------------------------------------------------------------------------- #
# Polyline generators for plotting (viz only — classification uses the analytic rules)
# --------------------------------------------------------------------------- #
def three_point_polyline(court: CourtSpec, n: int = 200) -> np.ndarray:
    """Ordered points tracing the 3-point boundary (corner line -> arc -> corner line)."""
    xc = court.corner_three_dist_m
    R = court.three_arc_radius_m
    yt = court.corner_transition_y_m
    theta = np.arctan2(yt, xc)  # angle where arc meets corner on the +x side
    arc_t = np.linspace(theta, np.pi - theta, n)
    arc = np.stack([R * np.cos(arc_t), R * np.sin(arc_t)], axis=1)
    yb = -court.rim_from_baseline_m  # painted corner-3 lines run all the way to the baseline
    right = np.array([[xc, yb], [xc, yt]])
    left = np.array([[-xc, yt], [-xc, yb]])
    return np.vstack([right, arc, left])


def paint_polygon(court: CourtSpec) -> np.ndarray:
    w = court.lane_half_width_m
    y0, y1 = -court.rim_from_baseline_m, court.ft_line_from_hoop_m  # lane runs baseline->FT
    return np.array([[-w, y0], [w, y0], [w, y1], [-w, y1], [-w, y0]])


FT_CIRCLE_RADIUS_M = 1.8288  # free-throw circle radius, 6 ft — identical in all specs


def landmark_points(court: CourtSpec) -> dict[str, np.ndarray]:
    """Canonical court landmarks in hoop-centred metres — the correspondences a user clicks
    for homography calibration. All derive from CourtSpec constants; more well-spread points
    is the A7 accuracy lever (skip any you cannot see crisply — 6 sharp beats 9 sloppy).
    Lane-space "blocks" are deliberately absent until their per-spec rulebook offsets are
    added with citations; the under-rim point is absent because it is not visually
    identifiable (the rim's image position comes from the rim-ellipse annotation instead)."""
    sx = court.sideline_x_m
    yb = -court.rim_from_baseline_m  # baseline is behind the hoop (negative Y)
    return {
        "baseline_left_corner": np.array([-sx, yb]),
        "baseline_right_corner": np.array([sx, yb]),
        "lane_baseline_left": np.array([-court.lane_half_width_m, yb]),
        "lane_baseline_right": np.array([court.lane_half_width_m, yb]),
        "ft_left": np.array([-court.lane_half_width_m, court.ft_line_from_hoop_m]),
        "ft_center": np.array([0.0, court.ft_line_from_hoop_m]),
        "ft_right": np.array([court.lane_half_width_m, court.ft_line_from_hoop_m]),
        "top_of_key": np.array([0.0, court.ft_line_from_hoop_m + FT_CIRCLE_RADIUS_M]),
        "three_apex": np.array([0.0, court.three_arc_radius_m]),
        # The crisp, painted reference on the corner-3 line is its baseline intersection.
        "corner_three_right": np.array([court.corner_three_dist_m, yb]),
        "corner_three_left": np.array([-court.corner_three_dist_m, yb]),
    }
