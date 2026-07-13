"""User-definable shot-zone partitions over court-plane coordinates.

Zones are a *view*, not a measurement: the pipeline stores every shot as a continuous
hoop-centred court position (metres), and a ZonePartition is a pure function from that
position to a category. Consequences that drive this module's design:

- Partitions are stored in **court coordinates**, never screen coordinates, so they are
  camera-independent: move the camera, recalibrate, and the same zones apply. A screen-drawn
  boundary is lifted through the session homography once (``lift_screen_polyline``) and kept
  in court space from then on.
- Re-bucketing is retroactive and free: applying a new partition to a session is a lookup
  over stored positions — no reprocessing of video.
- The "deep three" boundary is an **offset of the actual 3-point shape** (arc plus straight
  corner segments), not a larger circle: a pure radial threshold misclassifies the corner,
  where the line runs straight and the arc radius is never reached.
- Every boundary supports the on-the-line band: a shot within ``band_m`` of a boundary is
  flagged, because the location estimate carries error (see the A7 error model) and a
  category read within that error is a guess. ``compose_with_error_map`` turns the A7 error
  field into a per-boundary reliability score for the current camera placement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .court_model import CourtSpec, distance_to_three_line, get_court, three_point_polyline
from .homography import apply_homography

FEET = 0.3048  # metres per foot


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _point_polyline_distance(x: float, y: float, poly: np.ndarray) -> float:
    """Min distance from (x, y) to a polyline given as (N, 2) vertices."""
    p = np.array([x, y], dtype=float)
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom == 0.0, 1.0, denom)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon; ``poly`` is (N, 2), closed or open."""
    px, py = poly[:, 0], poly[:, 1]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if (py[i] > y) != (py[j] > y):
            x_cross = (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def offset_three_polyline(court: CourtSpec, offset_m: float, n: int = 240) -> np.ndarray:
    """Polyline of the 3-point shape offset outward by ``offset_m`` (arc R+d, corners x±(xc+d)).

    The transition where the straight corner segment meets the arc moves too:
    y_t' = sqrt((R+d)^2 - (xc+d)^2).
    """
    R = court.three_arc_radius_m + offset_m
    xc = court.corner_three_dist_m + offset_m
    yt = float(np.sqrt(max(R * R - xc * xc, 0.0)))
    theta = np.arctan2(yt, xc)
    arc_t = np.linspace(theta, np.pi - theta, n)
    arc = np.stack([R * np.cos(arc_t), R * np.sin(arc_t)], axis=1)
    right = np.array([[xc, 0.0], [xc, yt]])
    left = np.array([[-xc, yt], [-xc, 0.0]])
    return np.vstack([right, arc, left])


def distance_beyond_three(court: CourtSpec, x: float, y: float) -> float:
    """Signed offset from the 3-point shape: positive outside (three territory), negative
    inside. Uses the corner-vs-arc split of the actual shape."""
    yt = court.corner_transition_y_m
    if y <= yt:
        return abs(x) - court.corner_three_dist_m
    return float(np.hypot(x, y) - court.three_arc_radius_m)


def _circle_polyline(radius: float, y_min: float, n: int = 160) -> np.ndarray:
    """Arc of a hoop-centred circle clipped to y >= y_min (half-court side)."""
    if radius <= abs(y_min) or y_min <= -radius:
        t = np.linspace(0.0, np.pi, n)
    else:
        t0 = np.arcsin(np.clip(y_min / radius, -1.0, 1.0))
        t = np.linspace(t0, np.pi - t0, n)
    return np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)


# --------------------------------------------------------------------------- #
# Partition
# --------------------------------------------------------------------------- #
@dataclass
class ZonePartition:
    """A named partition of the court plane plus its boundary polylines.

    ``label_fn(x, y) -> str`` must return exactly one zone for every point of the half
    court. ``boundaries`` maps boundary names to court-space polylines (used for the
    on-the-line band, reliability scoring, and drawing).
    """

    name: str
    zones: tuple[str, ...]
    label_fn: Callable[[float, float], str]
    boundaries: dict[str, np.ndarray] = field(default_factory=dict)
    spec: dict = field(default_factory=dict)  # serializable construction record

    def label(self, x: float, y: float) -> str:
        return self.label_fn(float(x), float(y))

    def classify(self, x: float, y: float, *, band_m: float = 0.15) -> dict:
        zone = self.label(x, y)
        nearest, dist = None, float("inf")
        for bname, poly in self.boundaries.items():
            d = _point_polyline_distance(float(x), float(y), poly)
            if d < dist:
                nearest, dist = bname, d
        return {
            "zone": zone,
            "on_line": bool(dist <= band_m),
            "nearest_boundary": nearest,
            "dist_to_boundary_m": None if nearest is None else float(dist),
        }

    def rebucket(self, shots: Sequence[Mapping], *, band_m: float = 0.15) -> list[dict]:
        """Re-label stored shots (each with ``court_xy``) under this partition — the
        retroactive view change; no video reprocessing involved."""
        out = []
        for s in shots:
            x, y = s["court_xy"]
            out.append({**s, **self.classify(x, y, band_m=band_m), "partition": self.name})
        return out

    def to_dict(self) -> dict:
        return dict(self.spec) if self.spec else {"mode": "opaque", "name": self.name}


# --------------------------------------------------------------------------- #
# Presets (all parametric — distances in metres unless *_ft)
# --------------------------------------------------------------------------- #
def preset_basic3(
    court: CourtSpec | str = "nba", *, interior_radius_m: float = 7 * FEET
) -> ZonePartition:
    """interior (<= interior radius) / midrange / three. The user's basic taxonomy."""
    c = get_court(court)
    r_int = float(interior_radius_m)

    def lab(x: float, y: float) -> str:
        if distance_beyond_three(c, x, y) > 0:
            return "three"
        if np.hypot(x, y) <= r_int:
            return "interior"
        return "midrange"

    return ZonePartition(
        name="basic3",
        zones=("interior", "midrange", "three"),
        label_fn=lab,
        boundaries={
            "interior_arc": _circle_polyline(r_int, -c.rim_from_baseline_m),
            "three_line": three_point_polyline(c),
        },
        spec={"mode": "basic3", "court": c.name, "interior_radius_m": r_int},
    )


def preset_extended(
    court: CourtSpec | str = "nba",
    *,
    interior_radius_m: float = 7 * FEET,
    mid_split_radius_m: float = 5.2,
    deep_three_offset_m: float = 0.9,
) -> ZonePartition:
    """interior / short-mid / long-mid / three / deep-three (offset of the true 3PT shape)."""
    c = get_court(court)
    r_int, r_mid, d3 = float(interior_radius_m), float(mid_split_radius_m), float(deep_three_offset_m)

    def lab(x: float, y: float) -> str:
        beyond = distance_beyond_three(c, x, y)
        if beyond > d3:
            return "deep-three"
        if beyond > 0:
            return "three"
        r = np.hypot(x, y)
        if r <= r_int:
            return "interior"
        return "short-mid" if r <= r_mid else "long-mid"

    return ZonePartition(
        name="extended",
        zones=("interior", "short-mid", "long-mid", "three", "deep-three"),
        label_fn=lab,
        boundaries={
            "interior_arc": _circle_polyline(r_int, -c.rim_from_baseline_m),
            "mid_split_arc": _circle_polyline(r_mid, -c.rim_from_baseline_m),
            "three_line": three_point_polyline(c),
            "deep_three_line": offset_three_polyline(c, d3),
        },
        spec={
            "mode": "extended", "court": c.name, "interior_radius_m": r_int,
            "mid_split_radius_m": r_mid, "deep_three_offset_m": d3,
        },
    )


def preset_spots(
    court: CourtSpec | str = "nba",
    *,
    interior_radius_m: float = 7 * FEET,
    corner_angle_deg: float = 27.0,
    wing_angle_deg: float = 65.0,
) -> ZonePartition:
    """The classic spot chart: interior, then {corner, wing, top} x {mid, three} per side
    collapsed to five named sectors x two ranges (11 zones). Sector angle is measured from
    the baseline direction (+x to the right), so small angles are corners."""
    c = get_court(court)
    r_int = float(interior_radius_m)
    a_c, a_w = np.deg2rad(corner_angle_deg), np.deg2rad(wing_angle_deg)

    def sector(x: float, y: float) -> str:
        theta = np.arctan2(max(y, 0.0), abs(x))  # 0 at baseline, pi/2 at top
        side = "right" if x >= 0 else "left"
        if theta < a_c:
            return f"{side}-corner"
        if theta < a_w:
            return f"{side}-wing"
        return "top"

    def lab(x: float, y: float) -> str:
        rng = "three" if distance_beyond_three(c, x, y) > 0 else "mid"
        if rng == "mid" and np.hypot(x, y) <= r_int:
            return "interior"
        return f"{sector(x, y)}-{rng}"

    zones = ["interior"] + [f"{s}-{r}" for s in
                            ("left-corner", "left-wing", "top", "right-wing", "right-corner")
                            for r in ("mid", "three")]
    # Sector boundary rays (drawn from interior arc outward for viz/band purposes).
    rays = {}
    r_out = c.three_arc_radius_m + 2.0
    for adeg, tag in ((corner_angle_deg, "corner"), (wing_angle_deg, "wing")):
        a = np.deg2rad(adeg)
        for sgn, side in ((1.0, "right"), (-1.0, "left")):
            rays[f"{side}_{tag}_ray"] = np.array(
                [[sgn * r_int * np.cos(a), r_int * np.sin(a)],
                 [sgn * r_out * np.cos(a), r_out * np.sin(a)]]
            )
    return ZonePartition(
        name="spots",
        zones=tuple(zones),
        label_fn=lab,
        boundaries={
            "interior_arc": _circle_polyline(r_int, -c.rim_from_baseline_m),
            "three_line": three_point_polyline(c),
            **rays,
        },
        spec={
            "mode": "spots", "court": c.name, "interior_radius_m": r_int,
            "corner_angle_deg": float(corner_angle_deg), "wing_angle_deg": float(wing_angle_deg),
        },
    )


def from_polygons(
    name: str, polygons: Mapping[str, np.ndarray | Sequence[Sequence[float]]],
    *, default_zone: str = "other",
) -> ZonePartition:
    """Freeform mode: zones from court-space polygons (first match wins, ``default_zone``
    elsewhere). Screen-drawn shapes should be lifted first via ``lift_screen_polyline``."""
    polys = {z: np.asarray(p, dtype=float) for z, p in polygons.items()}

    def lab(x: float, y: float) -> str:
        for z, poly in polys.items():
            if _point_in_polygon(x, y, poly):
                return z
        return default_zone

    boundaries = {f"{z}_outline": np.vstack([p, p[:1]]) for z, p in polys.items()}
    return ZonePartition(
        name=name,
        zones=tuple(list(polys) + [default_zone]),
        label_fn=lab,
        boundaries=boundaries,
        spec={"mode": "polygons", "name": name, "default_zone": default_zone,
              "polygons": {z: np.asarray(p, dtype=float).tolist() for z, p in polys.items()}},
    )


def from_dict(d: Mapping) -> ZonePartition:
    """Rebuild a partition from its serialized ``spec`` (YAML/JSON round-trip)."""
    mode = d.get("mode")
    if mode == "basic3":
        return preset_basic3(d.get("court", "nba"), interior_radius_m=d["interior_radius_m"])
    if mode == "extended":
        return preset_extended(
            d.get("court", "nba"), interior_radius_m=d["interior_radius_m"],
            mid_split_radius_m=d["mid_split_radius_m"], deep_three_offset_m=d["deep_three_offset_m"],
        )
    if mode == "spots":
        return preset_spots(
            d.get("court", "nba"), interior_radius_m=d["interior_radius_m"],
            corner_angle_deg=d["corner_angle_deg"], wing_angle_deg=d["wing_angle_deg"],
        )
    if mode == "polygons":
        return from_polygons(d.get("name", "custom"), d["polygons"],
                             default_zone=d.get("default_zone", "other"))
    raise ValueError(f"unknown partition mode {mode!r}")


# --------------------------------------------------------------------------- #
# Screen <-> court lifting and boundary reliability
# --------------------------------------------------------------------------- #
def lift_screen_polyline(
    H_img_from_court: np.ndarray, screen_pts: np.ndarray, *,
    court: CourtSpec | str = "nba", validate: bool = True,
) -> np.ndarray:
    """Lift a screen-drawn polyline to court coordinates through the session homography
    (inverted here: H maps court -> image, as estimated at calibration). With ``validate``,
    reject drawings that land outside a generous half-court box — the usual cause is a
    stale calibration or a drawing over non-floor pixels."""
    H_court_from_img = np.linalg.inv(np.asarray(H_img_from_court, dtype=float))
    pts = apply_homography(H_court_from_img, np.asarray(screen_pts, dtype=float))
    if validate:
        c = get_court(court)
        x_ok = np.all(np.abs(pts[:, 0]) <= c.sideline_x_m + 1.0)
        y_ok = np.all((pts[:, 1] >= -c.rim_from_baseline_m - 1.0) & (pts[:, 1] <= 16.0))
        if not (x_ok and y_ok):
            raise ValueError("lifted polyline falls outside the half court; "
                             "recalibrate or redraw on the floor region")
    return pts


def compose_with_error_map(
    partition: ZonePartition,
    err_fn: Callable[[float, float], float],
    *, band_m: float = 0.15, samples_per_boundary: int = 100,
) -> dict[str, dict]:
    """Score each boundary against a location-error field (e.g. the A7 Monte-Carlo P90
    field for the current camera placement): the fraction of the boundary where the
    location error exceeds the on-the-line band is the fraction where zone reads at the
    line are guesses. Returns {boundary: {reliable_fraction, mean_err_m, verdict}}."""
    out: dict[str, dict] = {}
    for bname, poly in partition.boundaries.items():
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        total = float(seg.sum())
        if total == 0.0:
            continue
        s = np.concatenate([[0.0], np.cumsum(seg)]) / total
        t = np.linspace(0.0, 1.0, samples_per_boundary)
        xs = np.interp(t, s, poly[:, 0])
        ys = np.interp(t, s, poly[:, 1])
        errs = np.array([float(err_fn(float(x), float(y))) for x, y in zip(xs, ys)])
        reliable = float(np.mean(errs <= band_m))
        out[bname] = {
            "reliable_fraction": reliable,
            "mean_err_m": float(errs.mean()),
            "verdict": "ok" if reliable >= 0.9 else ("marginal" if reliable >= 0.6 else "unreliable"),
        }
    return out
