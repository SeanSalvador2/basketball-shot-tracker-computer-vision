"""M4 detection-layer tests: bg-sub finds the ball on rendered frames, the torchvision
wrapper plumbing runs, TrackNet-lite trains + localizes, candidate fusion unions channels."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from bball.detect.bgsub import BgSubBallDetector, BgSubConfig, fuse_candidates
from bball.detect.interfaces import BallCandidate, Detection, iou
from bball.synth.camera import make_camera, project_trajectory
from bball.synth.physics import generate_shot
from bball.synth.render import SceneAppearance, render_clip
from bball.lift.court_model import get_court


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_detection_center():
    d = Detection(bbox=(10, 20, 30, 60), score=0.9, label="ball")
    assert np.allclose(d.center, [20, 40])
    assert d.wh == (20, 40)


def test_bgsub_finds_moving_ball_on_render():
    court = get_court("nba")
    cam = make_camera(azimuth_deg=90, height_m=2.5, distance_m=10.0)  # sideline: low occlusion
    shot = generate_shot(release_xy=(-3.0, 5.0), outcome="miss", miss_direction="long",
                         miss_magnitude_m=0.8, fps=60, pre_release_s=0.3, post_rim_s=0.2, seed=4)
    scale = 0.5
    out = render_clip(shot, cam, court, SceneAppearance(), scale=scale)
    frames = out["frames"]
    gt = out["ball_img_px"] * scale
    det = BgSubBallDetector(BgSubConfig(min_area=6, max_area=4000))
    per_frame = det.process_stream(frames, temporal=True)
    # In clean mid-flight frames the candidate should land near the GT ball.
    hits = 0
    checked = 0
    for i in range(8, shot.n_frames - 8):
        if np.isnan(gt[i]).any() or out["occlusion"][i] > 0.2:
            continue
        checked += 1
        if per_frame[i] and min(np.hypot(*(c.xy - gt[i])) for c in per_frame[i]) < 20:
            hits += 1
    assert checked > 5
    assert hits / checked > 0.5      # bg-sub recovers the ball on most clean-flight frames


def test_torchvision_detector_plumbing_runs():
    torch = pytest.importorskip("torch")
    from bball.detect.torchvision_detector import DetectorConfig, TorchvisionBallPlayerDetector

    # Random weights (no download); just validate the interface + class mapping plumbing.
    det = TorchvisionBallPlayerDetector(DetectorConfig(backbone="mobilenet", min_size=256, max_size=320),
                                        try_pretrained=False)
    frame = (np.random.rand(180, 320, 3) * 255).astype(np.uint8)
    dets = det.detect(frame, frame_idx=0)
    assert isinstance(dets, list)
    for d in dets:
        assert d.label in {"ball", "person"}


def test_tracknet_lite_trains_and_localizes():
    torch = pytest.importorskip("torch")
    from bball.detect.tracknet_lite import (
        TrackNetConfig,
        build_model,
        build_training_tensors,
        infer_ball,
        train_tracknet,
    )

    torch.manual_seed(0)
    torch.set_num_threads(1)  # determinism for the CI gate
    cfg = TrackNetConfig(in_frames=3, input_h=64, input_w=96, base_ch=8, sigma=2.5)
    H, W = 96, 144
    frames, centers = [], []
    rng = np.random.default_rng(0)
    cx, cy = 30.0, 40.0
    for i in range(14):
        img = np.zeros((H, W), np.uint8)
        cx += 5.0
        cv2.circle(img, (int(cx), int(cy)), 5, 255, -1)
        frames.append(img)
        centers.append((cx, cy))
    clips = [{"frames_gray": frames, "centers_px": centers, "frame_hw": (H, W)}]
    X, Y = build_training_tensors(clips, cfg)
    model = build_model(cfg)
    losses = train_tracknet(model, X, Y, epochs=6, lr=2e-3, batch=8, seed=0)
    assert losses[-1] < losses[0]                       # it learns
    xy, score = infer_ball(model, frames, 10, cfg, (H, W), peak_thresh=0.1)
    assert xy is not None
    assert np.hypot(*(xy - np.array(centers[10]))) < 20  # localizes near the true ball


def test_fuse_candidates_unions_and_arbitrates():
    neural = [BallCandidate(frame_idx=0, xy=np.array([100.0, 100.0]), score=0.8, source="detector")]
    classical = [
        BallCandidate(frame_idx=0, xy=np.array([102.0, 101.0]), score=0.9, source="bgsub"),  # dup, higher
        BallCandidate(frame_idx=0, xy=np.array([400.0, 300.0]), score=0.6, source="bgsub"),  # new
    ]
    fused = fuse_candidates(neural, classical, merge_dist=20.0)
    assert len(fused) == 2
    assert fused[0].score == pytest.approx(0.9)          # detector box, bg-sub's higher score
