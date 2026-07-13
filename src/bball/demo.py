"""Demo helpers used by notebooks/demo.ipynb.

Loads the bundled synthetic clip + metadata, runs the full DETECT->TRACK->LIFT->CLASSIFY
pipeline shot by shot, and exposes the intermediate artifacts (detections, bridged track,
FSM outcome, lifted court position) so the notebook can visualize each stage. The pipeline is
real-footage-ready: the only synthetic-specific inputs are the clip and the calibration
homography/rim annotation, which for real footage come from the one-time session setup.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bball.events.fsm import ShotFSM
from bball.events.release import FlightSegmenter
from bball.lift.court_model import classify_with_band, get_court
from bball.lift.homography import apply_homography
from bball.lift.rim_frame import RimEllipse
from bball.pipeline import ShotResult, SessionReport, detect_ball_bgsub
from bball.track.ballistic import bridge_trajectory


def load_demo(assets_dir: str | Path):
    """Return (frames_bgr, meta, rim_ellipse, H_img_to_court, court)."""
    import cv2
    import imageio.v2 as imageio

    assets_dir = Path(assets_dir)
    with open(assets_dir / "demo_meta.json") as f:
        meta = json.load(f)
    reader = imageio.get_reader(str(assets_dir / "demo_clip.mp4"))
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in reader]
    reader.close()
    e = meta["rim_ellipse"]
    rim = RimEllipse(cx=e["cx"] * meta["scale"], cy=e["cy"] * meta["scale"],
                     a=e["a"] * meta["scale"], b=e["b"] * meta["scale"], theta_deg=e["theta_deg"])
    H_court_to_img = np.array(meta["H_court_to_img"])
    H_img_to_court = np.linalg.inv(H_court_to_img)
    court = get_court(meta["court_spec"])
    return frames, meta, rim, H_img_to_court, court


def process_shot(frames_slice, scale, rim_ellipse, fps=60.0):
    """Detect (bg-sub) -> bridge -> FSM on one shot's frames. Returns artifacts + outcome."""
    cands = detect_ball_bgsub(frames_slice, scale)
    times = np.arange(len(frames_slice)) / fps
    br = bridge_trajectory(cands, times, method="l1")
    fsm = ShotFSM(rim_ellipse)
    out = fsm.process_flight(br.xy, br.observed)
    return {"candidates": cands, "bridged": br, "outcome": out, "times": times}


def run_demo(assets_dir: str | Path):
    """Run the full pipeline over every shot in the demo clip; return a rich result dict."""
    frames, meta, rim, H_img_to_court, court = load_demo(assets_dir)
    scale = meta["scale"]
    per_shot = []
    report = SessionReport()
    for sm in meta["shots"]:
        fs = frames[sm["frame_start"]:sm["frame_end"]]
        art = process_shot(fs, scale, rim)
        out = art["outcome"]
        # LIFT: map the (ground-truth, in this synthetic demo) shooter position to court.
        feet_court = np.array(sm["gt_shooter_xy"])       # synthetic demo uses GT feet; Stage B tracks the shooter
        z = classify_with_band(court, feet_court[0], feet_court[1])
        res = ShotResult(outcome=out.outcome if out.outcome != "none" else "miss",
                         make_prob=out.make_prob, margin=out.margin_score,
                         release_t=0.0, rim_t=0.0, court_xy=tuple(feet_court.tolist()),
                         zone=z["zone"], on_line=z["on_line"])
        report.shots.append(res)
        per_shot.append({"meta": sm, "art": art, "result": res, "frames": fs})
    return {"frames": frames, "meta": meta, "rim": rim, "court": court,
            "H_img_to_court": H_img_to_court, "per_shot": per_shot, "report": report}


def accuracy_vs_gt(result: dict) -> dict:
    """Compare predicted outcomes to the demo's ground truth."""
    correct = sum(1 for ps in result["per_shot"] if ps["result"].outcome == ps["meta"]["gt_outcome"])
    n = len(result["per_shot"])
    return {"correct": correct, "n": n, "accuracy": correct / max(n, 1),
            "detail": [(ps["meta"]["gt_outcome"], ps["result"].outcome) for ps in result["per_shot"]]}
