"""A3 — detector input resolution {512, 768, 1088} (plan §7).

Runs the torchvision COCO-pretrained Faster R-CNN (mobilenet throughput backbone; a single
resnet50_fpn_v2 accuracy-reference point) at three `min_size` settings and measures small-ball
detection two ways:

  * Regime S (synthetic renders): ball recall @ IoU 0.3 + mAP@0.5 against renderer GT boxes.
  * Regime R (real, zero-shot PRELIMINARY): ball / person detection fire-rate + mean
    confidence on the two permissive HF real sets (emirsahin/basketball-ball images;
    ZhiChengAI/Basketball_V0 clips). No GT boxes ship in those exports, so the real signal is
    a *presence fire-rate*, not IoU-matched recall — labelled as such everywhere.

Weights: the COCO checkpoints download from download.pytorch.org (the block logged in the
pipeline report's D2 was transient; the two URLs are in docs/REPRODUCING.md). If the cache is
empty the detector falls back to random init, `pretrained=False` is recorded, and the numbers
are meaningless — the module flags that rather than pretending.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import REPO_ROOT, log_run, save_fig
from bball.detect.torchvision_detector import DetectorConfig, TorchvisionBallPlayerDetector
from bball.eval.metrics import average_precision_at_iou, detection_recall_at_iou
from bball.synth.camera import apparent_ball_radius_px, in_frame, make_camera, project_trajectory
from bball.synth.render import render_clip
from bball.synth.scenarios import generate_session, venue_scene

REAL_ZIP = REPO_ROOT / "data" / "external" / "hf-basketball-ball-mit" / "Basketball-Detection-1.zip"
REAL_CLIPS = REPO_ROOT / "data" / "external" / "hf-Basketball_V0" / "videos" / "1"


# --------------------------------------------------------------------------- #
# Regime S — synthetic renders with renderer GT ball boxes
# --------------------------------------------------------------------------- #
def _synthetic_frames(cfg: dict):
    az, h = cfg.get("azimuth_deg", 55), cfg.get("height_m", 1.5)
    n_shots = cfg.get("syn_shots", 4)
    stride = cfg.get("syn_frame_stride", 6)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=9.0)
    scene = venue_scene("gym_A", azimuth_deg=az, height_m=h)
    sess = generate_session(scene, n_shots=n_shots, fps=60, seed=cfg.get("seed", 20260713))
    frames, gt_boxes = [], []
    for shot in sess.shots:
        out = render_clip(shot, cam, scene.court, scene.appearance, scale=1.0)
        ball_px = out["ball_img_px"]
        rad = apparent_ball_radius_px(cam, shot.pos)
        for i in range(0, len(out["frames"]), stride):
            if np.isnan(ball_px[i]).any() or np.isnan(rad[i]):
                continue
            if not in_frame(cam, ball_px[i:i + 1])[0]:
                continue
            r = max(float(rad[i]), 3.0)
            cx, cy = ball_px[i]
            frames.append(out["frames"][i])
            gt_boxes.append([(cx - r, cy - r, cx + r, cy + r)])
    return frames, gt_boxes


def _run_synthetic(frames, gt_boxes, resolutions, backbone="mobilenet"):
    rows = []
    for res in resolutions:
        det = TorchvisionBallPlayerDetector(
            DetectorConfig(backbone=backbone, min_size=res, ball_score_thresh=0.02, score_thresh=0.3))
        pred_boxes, pred_scores = [], []
        for i, fr in enumerate(frames):
            balls = [d for d in det.detect(fr, i) if d.label == "ball"]
            pred_boxes.append([d.bbox for d in balls])
            pred_scores.append([d.score for d in balls])
        rec = detection_recall_at_iou(pred_boxes, gt_boxes, iou_thr=0.3)
        ap = average_precision_at_iou(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5)
        rows.append({"regime": "S", "backbone": backbone, "min_size": res,
                     "ball_recall_iou0.3": round(rec["recall"], 3), "mAP0.5": round(ap["ap"], 3),
                     "n_gt": rec["n_gt"], "pretrained": det.pretrained})
    return rows


# --------------------------------------------------------------------------- #
# Regime R — real images / clips, zero-shot fire-rate
# --------------------------------------------------------------------------- #
def _real_images(n: int):
    if not REAL_ZIP.exists():
        return []
    z = zipfile.ZipFile(REAL_ZIP)
    names = sorted(n2 for n2 in z.namelist() if n2.endswith(".jpg") and "/valid/" in n2)[:n]
    out = []
    for nm in names:
        buf = np.frombuffer(z.read(nm), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def _clip_frames(n_per_clip: int):
    frames = []
    if not REAL_CLIPS.exists():
        return frames
    for mp4 in sorted(REAL_CLIPS.glob("*.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 200
        idxs = np.linspace(total * 0.15, total * 0.85, n_per_clip).astype(int)
        want = set(int(i) for i in idxs)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(fr)
            i += 1
        cap.release()
    return frames


def _fire_stats(det, frames, ball_thr=0.3):
    nb = npp = 0
    ball_scores = []
    for i, fr in enumerate(frames):
        dets = det.detect(fr, i)
        balls = [d.score for d in dets if d.label == "ball" and d.score >= ball_thr]
        if balls:
            nb += 1
            ball_scores.append(max(balls))
        if any(d.label == "person" for d in dets):
            npp += 1
    n = max(len(frames), 1)
    return {"n": len(frames), "ball_fire_rate": round(nb / n, 3),
            "person_fire_rate": round(npp / n, 3),
            "ball_mean_conf": round(float(np.mean(ball_scores)), 3) if ball_scores else float("nan")}


def _run_real(resolutions, cfg):
    rows = []
    imgs = _real_images(cfg.get("real_images", 40))
    if imgs:
        for res in resolutions:
            det = TorchvisionBallPlayerDetector(
                DetectorConfig(backbone="mobilenet", min_size=res, ball_score_thresh=0.15, score_thresh=0.4))
            s = _fire_stats(det, imgs)
            rows.append({"regime": "R-zeroshot", "set": "emir-basketball-ball", "backbone": "mobilenet",
                         "min_size": res, **s, "note": "fire-rate, no GT boxes"})
        # one resnet50 accuracy-reference point on a small subset
        det = TorchvisionBallPlayerDetector(
            DetectorConfig(backbone="resnet50", min_size=768, ball_score_thresh=0.15, score_thresh=0.4))
        s = _fire_stats(det, imgs[:cfg.get("resnet_images", 12)])
        rows.append({"regime": "R-zeroshot", "set": "emir-basketball-ball", "backbone": "resnet50",
                     "min_size": 768, **s, "note": "fire-rate, no GT boxes"})
    clips = _clip_frames(cfg.get("clip_frames_per", 12))
    if clips:
        det = TorchvisionBallPlayerDetector(
            DetectorConfig(backbone="mobilenet", min_size=768, ball_score_thresh=0.15, score_thresh=0.4))
        s = _fire_stats(det, clips)
        rows.append({"regime": "R-zeroshot", "set": "Basketball_V0-clips", "backbone": "mobilenet",
                     "min_size": 768, **s, "note": "fire-rate, no GT boxes"})
    return rows


def _plot(syn_rows, real_rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    res = [r["min_size"] for r in syn_rows]
    ax[0].plot(res, [r["ball_recall_iou0.3"] for r in syn_rows], "o-", label="recall@IoU0.3")
    ax[0].plot(res, [r["mAP0.5"] for r in syn_rows], "s--", label="mAP@0.5")
    ax[0].set_title("Regime S: synthetic renders (zero-shot COCO)")
    ax[0].set_xlabel("min_size (px)")
    ax[0].set_ylabel("score")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=8)
    emir = [r for r in real_rows if r.get("set") == "emir-basketball-ball" and r["backbone"] == "mobilenet"]
    if emir:
        res2 = [r["min_size"] for r in emir]
        ax[1].plot(res2, [r["ball_fire_rate"] for r in emir], "o-", label="ball fire-rate")
        ax[1].plot(res2, [r["ball_mean_conf"] for r in emir], "^--", label="ball mean conf")
        ax[1].plot(res2, [r["person_fire_rate"] for r in emir], "s:", label="person fire-rate")
        ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Regime R (zero-shot, prelim): real emir-shoots frames")
    ax[1].set_xlabel("min_size (px)")
    ax[1].set_ylabel("rate / confidence")
    ax[1].legend(fontsize=8)
    fig.suptitle("A3 — detector input resolution (S: synthetic GT; R: real fire-rate, no GT)")
    fig.tight_layout()
    return fig


def run(cfg: dict) -> dict:
    import torch

    torch.set_num_threads(cfg.get("torch_threads", 4))
    resolutions = cfg.get("resolutions", [512, 768, 1088])
    seed = cfg.get("seed", 20260713)

    frames, gt_boxes = _synthetic_frames(cfg)
    syn_rows = _run_synthetic(frames, gt_boxes, resolutions)
    real_rows = _run_real(resolutions, cfg)
    rows = syn_rows + real_rows

    fig = _plot(syn_rows, real_rows)
    fig_path = save_fig(fig, "a3_resolution")

    metrics = {}
    for r in syn_rows:
        metrics[f"S_recall_{r['min_size']}"] = r["ball_recall_iou0.3"]
        metrics[f"S_mAP_{r['min_size']}"] = r["mAP0.5"]
    for r in real_rows:
        if r.get("set") == "emir-basketball-ball" and r["backbone"] == "mobilenet":
            metrics[f"R_ballfire_{r['min_size']}"] = r["ball_fire_rate"]
    pretrained = bool(syn_rows[0]["pretrained"]) if syn_rows else False
    run_id = log_run("bball-A3", "a3_resolution",
                     params={"seed": seed, "resolutions": resolutions, "backbone": "mobilenet(+resnet50 ref)",
                             "regime": "S synthetic-GT + R zero-shot fire-rate", "pretrained": pretrained,
                             "n_synth_frames": len(frames)},
                     metrics=metrics, figures={"a3": fig_path}, summary_rows=rows)
    print(f"[A3] run_id={run_id} pretrained={pretrained} n_synth={len(frames)}")
    for r in rows:
        print("   ", r)
    return {"run_id": run_id, "rows": rows}
