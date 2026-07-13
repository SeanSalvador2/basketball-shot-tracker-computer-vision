"""SYNTH — the synthetic-data experiment engine (Stage A's workhorse).

physics.py    3D ballistic shot generator (grounded in real basketball ranges)
camera.py     scene camera over lift.projection (azimuth/height/distance/FOV)
render.py     motion-blurred ball over a procedural court + detection-noise model
scenarios.py  scene/venue bundles + session generator + players + hard negatives
build_bundle  the reproducible bundle CLI (make synth)

Outputs BOTH ground-truth trajectories/events (logic experiments) and rendered mp4 clips
(end-to-end demos). Every physical constant is cited (plan gate G4).
"""
from bball.synth.camera import apparent_ball_radius_px, make_camera, project_trajectory
from bball.synth.physics import Shot, generate_shot, solve_launch
from bball.synth.render import (
    DetectionNoiseModel,
    SceneAppearance,
    compute_rim_image_geometry,
    occlusion_fraction,
    render_clip,
    write_mp4,
)
from bball.synth.scenarios import (
    Session,
    SceneConfig,
    generate_session,
    generate_lob_pass,
    simulate_players,
    venue_scene,
)

__all__ = [
    "Shot", "generate_shot", "solve_launch",
    "make_camera", "project_trajectory", "apparent_ball_radius_px",
    "SceneAppearance", "DetectionNoiseModel", "compute_rim_image_geometry",
    "occlusion_fraction", "render_clip", "write_mp4",
    "SceneConfig", "Session", "venue_scene", "generate_session",
    "generate_lob_pass", "simulate_players",
]
