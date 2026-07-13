"""Build the synthetic session bundle (Stage A's dataset).

A bundle is a set of sessions across distinct scene configs (venues x camera placements),
mirroring the collection grid. It is fully regenerable from (config, seed) — gate G5 — so
the heavy artifacts stay gitignored under data/synthetic/ while the config is committed.

    python -m bball.synth.build_bundle --config configs/synth_bundle.yaml

`generate_bundle` returns sessions in memory (ablations/EDA call it directly);
`save_bundle` writes a manifest + per-session npz + events JSON for the demo/EDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bball.synth.camera import make_camera, project_trajectory, apparent_ball_radius_px
from bball.synth.render import DetectionNoiseModel, compute_rim_image_geometry, occlusion_fraction
from bball.synth.scenarios import Session, generate_session, venue_scene
from bball.utils.config import config_hash, load_config
from bball.utils.logging import get_logger

log = get_logger("bball.synth.build_bundle")


def _shot_detection_stream(shot, camera, rim_geom, noise: DetectionNoiseModel, rng):
    ball_img = project_trajectory(camera, shot.pos)
    ball_rad = apparent_ball_radius_px(camera, shot.pos)
    occl = np.array([
        occlusion_fraction(None if np.isnan(ball_img[i]).any() else ball_img[i],
                           float(ball_rad[i]) if not np.isnan(ball_rad[i]) else 3.0, rim_geom)
        for i in range(shot.n_frames)
    ])
    stream = noise.stream(ball_img, ball_rad, occl, rng)
    return ball_img, ball_rad, occl, stream


def generate_bundle(cfg: dict) -> dict:
    """Return {scene_id: {'session': Session, 'camera': Camera, 'rim_geom':..., 'streams':...}}."""
    seed = int(cfg.get("seed", 1729))
    fps = float(cfg.get("fps", 60.0))
    n_shots = int(cfg.get("n_shots", 40))
    noise = DetectionNoiseModel(**cfg.get("detection_noise", {}))
    out = {}
    for i, sc in enumerate(cfg["scenes"]):
        scene = venue_scene(sc["venue"], azimuth_deg=sc.get("azimuth_deg", 45.0),
                            height_m=sc.get("height_m", 3.0), distance_m=sc.get("distance_m", 9.0),
                            court_spec=cfg.get("court_spec", "nba"))
        sess = generate_session(scene, n_shots=n_shots, fps=fps, seed=seed + i)
        camera = make_camera(width_px=scene.width_px, height_px=scene.height_px,
                             hfov_deg=scene.hfov_deg, azimuth_deg=scene.azimuth_deg,
                             height_m=scene.height_m, distance_m=scene.distance_m)
        rim_geom = compute_rim_image_geometry(camera, (0.0, 0.0))
        rng = np.random.default_rng(seed + 1000 + i)
        streams = []
        for shot in sess.shots:
            streams.append(_shot_detection_stream(shot, camera, rim_geom, noise, rng))
        out[scene.scene_id] = {"session": sess, "camera": camera, "rim_geom": rim_geom,
                               "streams": streams, "scene": scene}
        log.info("scene %s: %d shots, %d negatives", scene.scene_id, len(sess.shots), len(sess.negatives))
    out["_meta"] = {"config_hash": config_hash(cfg), "seed": seed, "fps": fps, "n_scenes": len(cfg["scenes"])}
    return out


def save_bundle(bundle: dict, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"scenes": [], "meta": bundle.get("_meta", {})}
    for scene_id, data in bundle.items():
        if scene_id == "_meta":
            continue
        sess: Session = data["session"]
        # Pack positions with a lengths index.
        lengths = np.array([s.n_frames for s in sess.shots])
        pos = np.concatenate([s.pos for s in sess.shots], axis=0)
        times = np.concatenate([s.t for s in sess.shots], axis=0)
        np.savez_compressed(outdir / f"{scene_id}.npz", pos=pos, times=times, lengths=lengths)
        with open(outdir / f"{scene_id}_events.json", "w") as f:
            json.dump(sess.events_table(), f, indent=2)
        manifest["scenes"].append({
            "scene_id": scene_id, "venue": data["scene"].venue,
            "azimuth_deg": data["scene"].azimuth_deg, "height_m": data["scene"].height_m,
            "n_shots": len(sess.shots),
        })
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("saved bundle -> %s (%d scenes)", outdir, len(manifest["scenes"]))
    return outdir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the synthetic session bundle")
    ap.add_argument("--config", default="configs/synth_bundle.yaml")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    bundle = generate_bundle(cfg)
    if not args.no_save:
        save_bundle(bundle, cfg.get("output_dir", "data/synthetic") + "/" + cfg.get("name", "bundle"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
