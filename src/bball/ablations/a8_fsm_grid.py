"""A8 — FSM parameter grid (make_fraction x confirm_frames) -> F1 sensitivity surface.

Hypothesis: a plateau exists (robustness), not a knife-edge (an overfit rule); we report the
sensitivity surface, not just the optimum. Regime: S (+ R events in Stage B).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.events.fsm import FSMConfig, ShotFSM
from bball.eval.metrics import outcome_prf
from bball.synth.camera import make_camera
from bball.synth.render import compute_rim_image_geometry
from bball.synth.scenarios import venue_scene, generate_session


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 20260713)
    make_fracs = cfg.get("make_fractions", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    confirms = cfg.get("confirm_frames", [1, 2, 3, 4, 5, 6])
    az, h = cfg.get("azimuth_deg", 55), cfg.get("height_m", 1.5)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=9.0)
    ell = compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse
    # tune on several sessions to reduce variance
    sessions = [generate_session(venue_scene("gym_A", azimuth_deg=az, height_m=h),
                                 n_shots=cfg.get("n_shots", 50), fps=60, seed=seed + k) for k in range(3)]
    imgs = [[cam.project(s.pos) for s in sess.shots] for sess in sessions]

    surface = np.zeros((len(make_fracs), len(confirms)))
    rows = []
    for i, mf in enumerate(make_fracs):
        for j, cfr in enumerate(confirms):
            fsm = ShotFSM(ell, FSMConfig(make_fraction=mf, confirm_frames=cfr))
            preds, gts = [], []
            for sess, sess_imgs in zip(sessions, imgs):
                for shot, img in zip(sess.shots, sess_imgs):
                    out = fsm.process_flight(img)
                    if out.outcome == "none":
                        continue
                    preds.append(out.outcome)
                    gts.append(shot.outcome)
            f1 = outcome_prf(preds, gts).f1
            surface[i, j] = f1
            rows.append({"make_fraction": mf, "confirm_frames": cfr, "f1": round(f1, 3)})

    fig = _plot(surface, make_fracs, confirms)
    fig_path = save_fig(fig, "a8_fsm_sensitivity")
    best = np.unravel_index(np.argmax(surface), surface.shape)
    metrics = {"best_f1": float(surface[best]), "best_make_fraction": make_fracs[best[0]],
               "best_confirm_frames": confirms[best[1]],
               "f1_range": float(surface.max() - surface[surface > 0].min())}
    run_id = log_run("bball-A8", "a8_fsm_grid", params={"make_fractions": make_fracs,
                     "confirm_frames": confirms, "seed": seed}, metrics=metrics,
                     figures={"a8": fig_path}, summary_rows=rows)
    print(f"[A8] run_id={run_id} best_f1={surface[best]:.3f} at mf={make_fracs[best[0]]} confirm={confirms[best[1]]}")
    return {"run_id": run_id, "rows": rows}


def _plot(surface, make_fracs, confirms):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(surface, origin="lower", aspect="auto", cmap="viridis",
                   extent=[confirms[0] - 0.5, confirms[-1] + 0.5, make_fracs[0] - 0.05, make_fracs[-1] + 0.05])
    for i, mf in enumerate(make_fracs):
        for j, cf in enumerate(confirms):
            ax.text(cf, mf, f"{surface[i, j]:.2f}", ha="center", va="center",
                    color="w" if surface[i, j] < 0.8 else "k", fontsize=7)
    ax.set_xlabel("confirm_frames")
    ax.set_ylabel("make_fraction")
    ax.set_title("A8 — FSM F1 sensitivity surface (regime: S)")
    fig.colorbar(im, ax=ax, label="make/miss F1")
    return fig
