"""A1 — ball association, four arms (review R4):
  (1) per-frame bbox + ballistic bridging   (the primary path)
  (2) heatmap-temporal (TrackNet-lite, 3-frame)   [reduced scale — regime S]
  (3) bbox WITHOUT bridging
  (4) background-subtraction candidates fused with bbox

Hypothesis: at 20-40 px, bbox+bridging ~ heatmap on clean flight; heatmap wins only under
heavy blur/occlusion; bridging closes most of that gap at far lower cost; bg-sub fusion buys
recall on clean flight but fails under multi-mover chaos. Metrics: ball-track completeness
and downstream T2 F1. TrackNet-lite is trained at reduced scale on synthetic renders and is
labelled as such (regime S, reduced-scale). Configs scale up unchanged on a GPU (Stage B).
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.detect.bgsub import BgSubBallDetector, BgSubConfig, fuse_candidates
from bball.detect.interfaces import BallCandidate
from bball.detect.tracknet_lite import (
    TrackNetConfig,
    build_model,
    build_training_tensors,
    infer_ball,
    train_tracknet,
)
from bball.events.fsm import ShotFSM
from bball.eval.metrics import outcome_prf
from bball.synth.camera import apparent_ball_radius_px, make_camera, project_trajectory
from bball.synth.render import DetectionNoiseModel, SceneAppearance, compute_rim_image_geometry, occlusion_fraction, render_clip
from bball.synth.scenarios import venue_scene, generate_session
from bball.track.ballistic import bridge_trajectory


def _completeness(track_xy, gt_img, lo, hi, tol=25.0):
    """Fraction of in-frame FLIGHT frames whose track position is within tol px of GT. The
    flight window (release..rim-arrival) is the association metric; the non-ballistic net-drop
    after a make is excluded (no tracker is expected to follow the ball through the net)."""
    ok = tot = 0
    for i in range(lo, min(hi, len(gt_img))):
        if np.isnan(gt_img[i]).any():
            continue
        tot += 1
        if track_xy[i] is not None and not np.isnan(track_xy[i]).any() and np.hypot(*(track_xy[i] - gt_img[i])) < tol:
            ok += 1
    return ok / max(tot, 1)


def run(cfg: dict) -> dict:
    import torch

    torch.set_num_threads(cfg.get("torch_threads", 4))
    seed = cfg.get("seed", 20260713)
    az, h = cfg.get("azimuth_deg", 55), cfg.get("height_m", 1.5)
    n_shots = cfg.get("n_shots", 18)
    scale = cfg.get("render_scale", 0.4)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=9.0)
    scene = venue_scene("gym_A", azimuth_deg=az, height_m=h)
    rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
    fsm = ShotFSM(rim_geom.ellipse)
    sess = generate_session(scene, n_shots=n_shots, fps=60, seed=seed)

    # Render all shots once (reused across arms). Keep it small.
    clips = []
    for shot in sess.shots:
        out = render_clip(shot, cam, scene.court, scene.appearance, scale=scale)
        clips.append(out)

    # --- Train TrackNet-lite at reduced scale on the first half of the shots ---
    tcfg = TrackNetConfig(in_frames=3, input_h=cfg.get("tn_h", 96), input_w=cfg.get("tn_w", 160),
                          base_ch=cfg.get("tn_ch", 12), sigma=3.0)
    n_train = max(n_shots // 2, 4)
    train_clips = []
    for k in range(n_train):
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in clips[k]["frames"]]
        H, W = frames_gray[0].shape
        centers = [None if np.isnan(clips[k]["ball_img_px"][i] * scale).any() else clips[k]["ball_img_px"][i] * scale
                   for i in range(len(frames_gray))]
        train_clips.append({"frames_gray": frames_gray, "centers_px": centers, "frame_hw": (H, W)})
    X, Y = build_training_tensors(train_clips, tcfg)
    model = build_model(tcfg)
    losses = train_tracknet(model, X, Y, epochs=cfg.get("tn_epochs", 8), lr=1e-3, batch=16, seed=seed)

    noise = DetectionNoiseModel()
    bgcfg = BgSubConfig(min_area=4, max_area=4000)

    # Evaluate arms on the held-out second half.
    arms = {"bbox+bridge": [], "tracknet": [], "bbox_no_bridge": [], "bgsub_fusion": []}
    fsm_arms = {k: {"pred": [], "gt": []} for k in arms}
    for k in range(n_train, n_shots):
        shot = sess.shots[k]
        clip = clips[k]
        rel = int(round(shot.events["release_t"] * 60))           # flight window for completeness
        ra = int(round(shot.events["rim_arrival_t"] * 60)) + 3
        gt_img_full = project_trajectory(cam, shot.pos)           # native px
        gt_img_scaled = gt_img_full * scale
        rad = apparent_ball_radius_px(cam, shot.pos)
        occl = np.array([occlusion_fraction(None if np.isnan(gt_img_full[i]).any() else gt_img_full[i],
                                            float(rad[i]) if not np.isnan(rad[i]) else 3.0, rim_geom)
                         for i in range(shot.n_frames)])
        rng = np.random.default_rng(seed + k)
        det_stream = noise.stream(gt_img_full, rad, occl, rng)     # native px bbox arm

        # Arm 1: bbox + L1 bridging.
        br = bridge_trajectory(det_stream, shot.t, method="l1")
        arm1_xy = [br.xy[i] if not np.isnan(br.xy[i]).any() else None for i in range(len(br.xy))]
        arms["bbox+bridge"].append(_completeness(arm1_xy, gt_img_full, rel, ra))
        _fsm_record(fsm, br.xy, br.observed, shot, fsm_arms["bbox+bridge"])

        # Arm 3: bbox no bridge.
        arm3_xy = [c.xy if c.observed else None for c in det_stream]
        arms["bbox_no_bridge"].append(_completeness(arm3_xy, gt_img_full, rel, ra))
        xy3 = np.array([c.xy if c.observed else [np.nan, np.nan] for c in det_stream])
        _fsm_record(fsm, xy3, np.array([c.observed for c in det_stream]), shot, fsm_arms["bbox_no_bridge"])

        # Arm 2: TrackNet-lite (infer on rendered frames, scaled coords -> native).
        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in clip["frames"]]
        Hh, Ww = frames_gray[0].shape
        tn_xy = []
        for i in range(len(frames_gray)):
            xy, score = infer_ball(model, frames_gray, i, tcfg, (Hh, Ww), peak_thresh=0.25)
            tn_xy.append(None if xy is None else xy / scale)       # back to native px
        arms["tracknet"].append(_completeness(tn_xy, gt_img_full, rel, ra))
        tn_arr = np.array([p if p is not None else [np.nan, np.nan] for p in tn_xy])
        tn_obs = np.array([p is not None for p in tn_xy])
        tn_br = bridge_trajectory([BallCandidate(i, tn_xy[i], 0.8, 4.0) if tn_xy[i] is not None
                                   else BallCandidate(i, None) for i in range(len(tn_xy))], shot.t, method="l1")
        _fsm_record(fsm, tn_br.xy, tn_br.observed, shot, fsm_arms["tracknet"])

        # Arm 4: bg-sub fused with bbox (candidates in scaled px -> native).
        bg = BgSubBallDetector(bgcfg)
        bg_per_frame = bg.process_stream(clip["frames"], temporal=True)
        fused_xy = []
        for i in range(len(clip["frames"])):
            neural = [det_stream[i]] if det_stream[i].observed else []
            classical = [BallCandidate(i, c.xy / scale, c.score, c.radius_px / scale, source="bgsub")
                         for c in bg_per_frame[i] if c.xy is not None]
            fused = fuse_candidates(neural, classical)
            # pick candidate nearest GT (oracle association just for a completeness upper bound)
            best = None
            if fused:
                valid = [c for c in fused if c.xy is not None]
                if valid and not np.isnan(gt_img_full[i]).any():
                    best = min(valid, key=lambda c: np.hypot(*(c.xy - gt_img_full[i]))).xy
                elif valid:
                    best = valid[0].xy
            fused_xy.append(best)
        arms["bgsub_fusion"].append(_completeness(fused_xy, gt_img_full, rel, ra))
        fx = np.array([p if p is not None else [np.nan, np.nan] for p in fused_xy])
        fb = bridge_trajectory([BallCandidate(i, fused_xy[i], 0.8, 4.0) if fused_xy[i] is not None
                                else BallCandidate(i, None) for i in range(len(fused_xy))], shot.t, method="l1")
        _fsm_record(fsm, fb.xy, fb.observed, shot, fsm_arms["bgsub_fusion"])

    rows = []
    for arm in arms:
        comp = float(np.mean(arms[arm]))
        f1 = outcome_prf(fsm_arms[arm]["pred"], fsm_arms[arm]["gt"]).f1 if fsm_arms[arm]["pred"] else float("nan")
        rows.append({"arm": arm, "track_completeness": round(comp, 3), "t2_f1": round(f1, 3),
                     "n_eval": len(arms[arm])})

    fig = _plot(rows)
    fig_path = save_fig(fig, "a1_association_arms")
    metrics = {}
    for r in rows:
        metrics[f"completeness_{r['arm']}"] = r["track_completeness"]
        metrics[f"t2f1_{r['arm']}"] = r["t2_f1"]
    metrics["tracknet_final_loss"] = losses[-1]
    run_id = log_run("bball-A1", "a1_association", params={"seed": seed, "n_shots": n_shots,
                     "regime": "S (reduced-scale TrackNet)", "render_scale": scale},
                     metrics=metrics, figures={"a1": fig_path}, summary_rows=rows)
    print(f"[A1] run_id={run_id} " + " ".join(f"{r['arm']}:comp={r['track_completeness']},F1={r['t2_f1']}" for r in rows))
    return {"run_id": run_id, "rows": rows}


def _fsm_record(fsm, xy, observed, shot, bucket):
    out = fsm.process_flight(xy, observed)
    if out.outcome != "none":
        bucket["pred"].append(out.outcome)
        bucket["gt"].append(shot.outcome)


def _plot(rows):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    arms = [r["arm"] for r in rows]
    ax[0].bar(arms, [r["track_completeness"] for r in rows], color="tab:blue")
    ax[0].set_ylabel("ball-track completeness")
    ax[0].set_ylim(0, 1.02)
    ax[0].tick_params(axis="x", rotation=30, labelsize=8)
    ax[1].bar(arms, [r["t2_f1"] for r in rows], color="tab:green")
    ax[1].set_ylabel("downstream T2 F1")
    ax[1].set_ylim(0, 1.02)
    ax[1].tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("A1 — ball association arms (regime: S, TrackNet reduced-scale)")
    fig.tight_layout()
    return fig
