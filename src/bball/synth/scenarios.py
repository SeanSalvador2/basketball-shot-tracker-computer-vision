"""Scenarios — venue bundles, session generation, multi-agent players, hard negatives.

A **scene config** bundles court spec + camera placement + appearance (texture, lighting,
ball look). It is the unit of the synthetic split (plan §2.3): experiments split by scene
config exactly as real experiments split by session, so "memorize the background" is
impossible. Presets mirror the collection variation grid (§2.1): two indoor gyms, two
outdoor courts, several camera placements.

A **session** = one scene config + a shot script (zones x make/miss x miss-direction, with
rattle/dribble variants, plus negative blocks). `generate_session` returns fully-labelled
`Shot`s + an events table — the ground truth every metric is scored against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bball.lift.court_model import CourtSpec, get_court
from bball.synth.physics import RIM_HEIGHT_M, Shot, generate_shot, sample_release_params
from bball.synth.render import SceneAppearance


# --------------------------------------------------------------------------- #
# Scene config + venue presets
# --------------------------------------------------------------------------- #
@dataclass
class SceneConfig:
    scene_id: str
    court_spec: str = "nba"
    width_px: int = 1920
    height_px: int = 1080
    hfov_deg: float = 68.0
    azimuth_deg: float = 45.0
    height_m: float = 3.0
    distance_m: float = 9.0
    appearance: SceneAppearance = field(default_factory=SceneAppearance)
    venue: str = "gym_A"

    @property
    def court(self) -> CourtSpec:
        return get_court(self.court_spec)


_VENUE_APPEARANCE = {
    "gym_A": SceneAppearance(floor_bgr=(150, 180, 205), line_bgr=(245, 245, 245),
                             ball_bgr=(40, 110, 220), lighting_gain=1.0),
    "gym_B": SceneAppearance(floor_bgr=(120, 150, 175), line_bgr=(235, 235, 240),
                             ball_bgr=(45, 120, 225), lighting_gain=0.92,
                             lighting_tint=(1.05, 1.0, 0.95)),
    "outdoor_A": SceneAppearance(floor_bgr=(115, 120, 125), line_bgr=(220, 220, 220),
                                 ball_bgr=(35, 95, 200), lighting_gain=1.12,
                                 lighting_tint=(1.05, 1.02, 0.98)),
    "outdoor_B": SceneAppearance(floor_bgr=(95, 105, 110), line_bgr=(210, 215, 215),
                                 ball_bgr=(60, 130, 210), lighting_gain=0.8,
                                 lighting_tint=(0.95, 0.98, 1.1)),  # dusk-ish
}


def venue_scene(venue: str, *, azimuth_deg=45.0, height_m=3.0, distance_m=9.0,
                court_spec="nba", width_px=1920, height_px=1080, scene_id=None) -> SceneConfig:
    app = _VENUE_APPEARANCE.get(venue, SceneAppearance())
    sid = scene_id or f"{venue}_az{int(azimuth_deg)}_h{height_m:g}"
    return SceneConfig(scene_id=sid, court_spec=court_spec, width_px=width_px, height_px=height_px,
                       azimuth_deg=azimuth_deg, height_m=height_m, distance_m=distance_m,
                       appearance=app, venue=venue)


# --------------------------------------------------------------------------- #
# Shot script
# --------------------------------------------------------------------------- #
@dataclass
class ShotSpec:
    zone: str                 # short-range | midrange | 3PT
    side: str                 # left | center | right
    outcome: str              # make | miss
    miss_direction: str = "none"
    rattle: bool = False
    dribble: bool = False


ZONE_RADIAL = {"short-range": (0.6, 2.3), "midrange": (2.7, 6.3), "3PT": (0.35, 1.4)}
SIDE_ANGLE = {"left": -0.7, "center": 0.0, "right": 0.7}   # radians from +Y (into court)
MISS_DIRECTIONS = ["short", "long", "left", "right", "short-left", "long-right"]


def sample_shot_location(court: CourtSpec, zone: str, side: str, rng: np.random.Generator) -> np.ndarray:
    """Return shooter feet (court xy, hoop-centred) for a zone+side. 3PT radius is measured
    just outside the arc; angle spread comes from the side."""
    base_angle = SIDE_ANGLE[side] + rng.normal(0, 0.15)
    if zone == "3PT":
        r = court.three_arc_radius_m + rng.uniform(*ZONE_RADIAL["3PT"])
    else:
        lo, hi = ZONE_RADIAL[zone]
        r = rng.uniform(lo, hi)
    x = r * np.sin(base_angle)
    y = r * np.cos(base_angle)
    return np.array([float(x), float(y)])


def default_shot_script(rng: np.random.Generator, n: int = 60) -> list[ShotSpec]:
    """A balanced script: every zone x side, deliberate makes and misses (all miss
    directions), some rattle-ins and dribble pull-ups. Maker-biased slightly (~55%)."""
    zones = ["short-range", "midrange", "3PT"]
    sides = ["left", "center", "right"]
    specs: list[ShotSpec] = []
    for _ in range(n):
        zone = rng.choice(zones, p=[0.3, 0.35, 0.35])
        side = rng.choice(sides)
        make = rng.random() < 0.55
        outcome = "make" if make else "miss"
        miss_dir = "none" if make else str(rng.choice(MISS_DIRECTIONS))
        rattle = make and rng.random() < 0.15
        dribble = rng.random() < 0.4
        specs.append(ShotSpec(zone=zone, side=side, outcome=outcome,
                              miss_direction=miss_dir, rattle=rattle, dribble=dribble))
    return specs


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    scene: SceneConfig
    shots: list[Shot]
    negatives: list[Shot] = field(default_factory=list)
    players: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def events_table(self) -> list[dict]:
        rows = []
        for i, s in enumerate(self.shots):
            rows.append({
                "shot_idx": i, "outcome": s.outcome, "miss_direction": s.miss_direction,
                "release_xy": s.release_xy.tolist(), "release_t": s.events["release_t"],
                "rim_arrival_t": s.events["rim_arrival_t"], "apex_height_m": s.apex_height_m,
                "release_speed": s.release_speed, "release_angle_deg": s.release_angle_deg,
                "dribble": s.meta.get("dribble", False), "rattle": s.meta.get("rattle", False),
            })
        return rows


def generate_session(scene: SceneConfig, *, n_shots: int = 60, fps: float = 60.0,
                     seed: int = 0, script: list[ShotSpec] | None = None,
                     n_negatives: int = 4) -> Session:
    rng = np.random.default_rng(seed)
    court = scene.court
    script = script or default_shot_script(rng, n=n_shots)
    shots: list[Shot] = []
    for spec in script:
        loc = sample_shot_location(court, spec.zone, spec.side, rng)
        params = sample_release_params(rng)
        shot = generate_shot(
            release_xy=loc, hoop_ground_xy=(0.0, 0.0), outcome=spec.outcome,
            miss_direction=spec.miss_direction, miss_magnitude_m=float(rng.uniform(0.35, 0.9)),
            rattle=spec.rattle, dribble=spec.dribble, fps=fps,
            seed=int(rng.integers(1 << 31)), **params)
        shot.meta["zone"] = spec.zone
        shot.meta["side"] = spec.side
        shots.append(shot)
    negatives = [generate_lob_pass(court, fps=fps, seed=int(rng.integers(1 << 31)))
                 for _ in range(n_negatives)]
    return Session(scene=scene, shots=shots, negatives=negatives,
                   meta={"seed": seed, "n_shots": len(shots), "fps": fps})


# --------------------------------------------------------------------------- #
# Hard negatives + players
# --------------------------------------------------------------------------- #
def generate_lob_pass(court: CourtSpec, *, fps: float = 60.0, seed: int = 0) -> Shot:
    """A lob pass that arcs *near* the rim region but is caught — the adversarial near-
    positive for T1 (review R3/R7). Labelled outcome='negative'; the FSM must not count it."""
    rng = np.random.default_rng(seed)
    passer = np.array([rng.uniform(-4, 4), rng.uniform(5, 8)])
    receiver = np.array([rng.uniform(-2, 2), rng.uniform(1.5, 3.0)])  # near the hoop, not through
    # Aim to pass over a point beside the rim at a lower apex than a shot.
    target = receiver
    shot = generate_shot(
        release_xy=passer, hoop_ground_xy=tuple(receiver), outcome="make",
        release_angle_deg=float(rng.uniform(28, 38)),  # flatter than a shot
        release_height_m=float(rng.uniform(1.6, 2.0)), fps=fps, post_rim_s=0.4,
        seed=int(rng.integers(1 << 31)))
    shot.outcome = "miss"  # placeholder; overwritten below
    shot.meta["is_negative"] = True
    shot.meta["kind"] = "lob_pass"
    shot.outcome = "negative"  # type: ignore
    shot.miss_direction = "none"  # type: ignore
    return shot


def simulate_players(n: int, n_frames: int, court: CourtSpec, *, fps: float = 60.0,
                     seed: int = 0) -> list[dict]:
    """Waypoint walkers: each player drifts between random court points at walking speed,
    producing (n_frames, 2) ground tracks for bbox projection (A4 tracker tests)."""
    rng = np.random.default_rng(seed)
    players = []
    dt = 1.0 / fps
    speed = 1.6  # m/s walking
    for pid in range(n):
        pos = np.array([rng.uniform(-4, 4), rng.uniform(1.0, 8.0)])
        track = np.zeros((n_frames, 2))
        waypoint = np.array([rng.uniform(-5, 5), rng.uniform(0.5, 9.0)])
        for i in range(n_frames):
            to_wp = waypoint - pos
            d = np.linalg.norm(to_wp)
            if d < 0.3:
                waypoint = np.array([rng.uniform(-5, 5), rng.uniform(0.5, 9.0)])
                to_wp = waypoint - pos
                d = np.linalg.norm(to_wp)
            pos = pos + (to_wp / (d + 1e-9)) * speed * dt
            track[i] = pos
        players.append({"player_id": pid, "pos_xy": track, "height_m": float(rng.uniform(1.75, 2.0))})
    return players
