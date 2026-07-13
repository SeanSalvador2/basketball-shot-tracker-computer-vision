"""A5 — occlusion bridging OFF vs Level-1 vs Level-2-informed x gap length {3,8,15,30}.

Hypothesis: without bridging, T2 (make/miss) F1 collapses beyond ~8-frame gaps (rim-ball
occlusion is longer); Level-1 image-space bridging degrades gracefully to ~30 frames;
Level-2 anchoring adds little for T2 (it matters for T5). Regime: S.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.detect.interfaces import BallCandidate
from bball.events.fsm import ShotFSM
from bball.eval.metrics import outcome_prf
from bball.lift.rim_frame import rim_3d_center
from bball.synth.camera import apparent_ball_radius_px, make_camera
from bball.synth.render import compute_rim_image_geometry
from bball.synth.scenarios import venue_scene, generate_session
from bball.track.ballistic import bridge_trajectory


def _stream_with_gap(cam, shot, gap_len, rng, jitter_px=1.5):
    img = cam.project(shot.pos)
    rad = apparent_ball_radius_px(cam, shot.pos)
    ra = int(np.argmin(np.abs(shot.t - shot.events["rim_arrival_t"])))
    g0 = max(ra - gap_len // 2, 1)
    g1 = min(g0 + gap_len, shot.n_frames)
    cands = []
    for i in range(shot.n_frames):
        if np.isnan(img[i]).any() or (g0 <= i < g1):
            cands.append(BallCandidate(i, None))
        else:
            xy = img[i] + rng.normal(0, jitter_px, size=2)
            cands.append(BallCandidate(i, xy, 0.9, float(rad[i]) if not np.isnan(rad[i]) else 3.0))
    return cands


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 20260713)
    gaps = cfg.get("gap_lengths", [3, 8, 15, 30])
    methods = cfg.get("methods", ["off", "l1", "l2"])
    az, h, d = cfg.get("azimuth_deg", 55), cfg.get("height_m", 1.5), cfg.get("distance_m", 9.0)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=d)
    rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
    fsm = ShotFSM(rim_geom.ellipse)
    sess = generate_session(venue_scene("gym_A", azimuth_deg=az, height_m=h), n_shots=cfg.get("n_shots", 60),
                            fps=60, seed=seed)

    rows = []
    curves = {m: {} for m in methods}
    for method in methods:
        for gap in gaps:
            preds, gts = [], []
            for si, shot in enumerate(sess.shots):
                rng = np.random.default_rng(seed + si + gap)
                cands = _stream_with_gap(cam, shot, gap, rng)
                rk = {"shooter_feet_xy": shot.release_xy, "rim_center_3d": rim_3d_center((0.0, 0.0))}
                br = bridge_trajectory(cands, shot.t, method=method, camera=cam if method == "l2" else None,
                                       reconstruct_kwargs=rk if method == "l2" else None)
                out = fsm.process_flight(br.xy, br.observed)
                if out.outcome == "none":
                    continue
                preds.append(out.outcome)
                gts.append(shot.outcome)
            prf = outcome_prf(preds, gts)
            curves[method][gap] = prf.f1
            rows.append({"method": method, "gap_len": gap, "f1": round(prf.f1, 3),
                         "n": len(preds), "tp": prf.tp, "fp": prf.fp, "fn": prf.fn})

    fig = _plot(curves, gaps, methods)
    fig_path = save_fig(fig, "a5_bridging_gap")
    metrics = {f"f1_{m}_gap{g}": curves[m][g] for m in methods for g in gaps}
    run_id = log_run("bball-A5", "a5_bridging", params={"gaps": gaps, "methods": methods, "seed": seed},
                     metrics=metrics, figures={"a5": fig_path}, summary_rows=rows)
    print(f"[A5] run_id={run_id} figure={fig_path.name}")
    return {"run_id": run_id, "rows": rows}


def _plot(curves, gaps, methods):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"off": "no bridging", "l1": "Level-1 (image-space)", "l2": "Level-2 (3D-anchored)"}
    for m in methods:
        ax.plot(gaps, [curves[m][g] for g in gaps], "o-", label=labels.get(m, m))
    ax.set_xlabel("occlusion gap length (frames)")
    ax.set_ylabel("T2 make/miss F1")
    ax.set_ylim(0, 1.02)
    ax.set_title("A5 — occlusion bridging vs gap length (regime: S)")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig
