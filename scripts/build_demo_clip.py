#!/usr/bin/env python3
"""Render the bundled synthetic demo clip for notebooks/demo.ipynb.

Produces notebooks/assets/demo_clip.mp4 (a short multi-shot session, the one committed mp4 —
an explicit exception to the .gitignore) plus notebooks/assets/demo_meta.json carrying the
camera, rim ellipse, per-shot ground truth (shooter position, outcome, timing) and the
image->court homography, so the notebook can run the full pipeline and score itself.

    python scripts/build_demo_clip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bball.lift.rim_frame import RimEllipse  # noqa: E402
from bball.synth.camera import make_camera  # noqa: E402
from bball.synth.render import compute_rim_image_geometry, render_clip  # noqa: E402
from bball.synth.scenarios import ShotSpec, default_shot_script, venue_scene, sample_shot_location  # noqa: E402
from bball.synth.physics import generate_shot, sample_release_params  # noqa: E402

ASSETS = REPO / "notebooks" / "assets"


def build(seed: int = 20260713, n_shots: int = 6, scale: float = 0.5,
          azimuth_deg: float = 55, height_m: float = 1.5) -> None:
    rng = np.random.default_rng(seed)
    scene = venue_scene("gym_A", azimuth_deg=azimuth_deg, height_m=height_m)
    cam = make_camera(azimuth_deg=azimuth_deg, height_m=height_m, distance_m=9.0)
    rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
    court = scene.court

    # A curated, legible script: makes and misses across zones (deterministic outcomes).
    specs = [
        ShotSpec("3PT", "center", "make"),
        ShotSpec("midrange", "left", "miss", "left"),
        ShotSpec("short-range", "center", "make"),
        ShotSpec("3PT", "right", "miss", "long"),
        ShotSpec("midrange", "center", "make", rattle=True),
        ShotSpec("3PT", "left", "miss", "right"),
    ][:n_shots]

    all_frames = []
    meta_shots = []
    for spec in specs:
        loc = sample_shot_location(court, spec.zone, spec.side, rng)
        params = sample_release_params(rng)
        shot = generate_shot(release_xy=loc, hoop_ground_xy=(0.0, 0.0), outcome=spec.outcome,
                             miss_direction=spec.miss_direction, miss_magnitude_m=0.6,
                             rattle=spec.rattle, fps=60, seed=int(rng.integers(1 << 31)), **params)
        out = render_clip(shot, cam, court, scene.appearance, scale=scale)
        start = len(all_frames)
        all_frames.extend(out["frames"])
        meta_shots.append({
            "frame_start": start, "frame_end": len(all_frames),
            "gt_outcome": shot.outcome, "gt_zone": spec.zone,
            "gt_shooter_xy": shot.release_xy.tolist(),
            "gt_release_t": shot.events["release_t"], "gt_rim_t": shot.events["rim_arrival_t"],
        })

    ASSETS.mkdir(parents=True, exist_ok=True)
    from bball.synth.render import write_mp4

    write_mp4(all_frames, str(ASSETS / "demo_clip.mp4"), fps=60)

    ell = rim_geom.ellipse
    H_court_to_img = cam.ground_homography()
    meta = {
        "fps": 60, "scale": scale, "n_frames": len(all_frames),
        "camera": {"azimuth_deg": azimuth_deg, "height_m": height_m, "distance_m": 9.0,
                   "width_px": cam.width_px, "height_px": cam.height_px},
        "rim_ellipse": {"cx": ell.cx, "cy": ell.cy, "a": ell.a, "b": ell.b, "theta_deg": ell.theta_deg},
        "H_court_to_img": H_court_to_img.tolist(),
        "court_spec": scene.court_spec, "shots": meta_shots,
    }
    with open(ASSETS / "demo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    size_mb = (ASSETS / "demo_clip.mp4").stat().st_size / 1e6
    print(f"[demo] wrote {ASSETS/'demo_clip.mp4'} ({size_mb:.1f} MB, {len(all_frames)} frames) + demo_meta.json")


if __name__ == "__main__":
    build()
