"""A9 — calibration: none vs temperature vs Platt, per data regime.

Hypothesis: temperature scaling cuts ECE >= 50% at zero accuracy cost (margins are
monotone-informative). Leakage discipline (R6): calibrators are fit on val-cal, ECE/Brier
and the reliability diagram are reported on the held-out test sessions. Regime: S.
"""
from __future__ import annotations

import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.events.calibration import (
    PlattScaler,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
)
from bball.events.fsm import ShotFSM
from bball.synth.camera import make_camera
from bball.synth.render import compute_rim_image_geometry
from bball.synth.scenarios import venue_scene, generate_session
from bball.viz.reliability import reliability_diagram


def _collect(cam, ell, session):
    fsm = ShotFSM(ell)
    margins, labels = [], []
    for shot in session.shots:
        out = fsm.process_flight(cam.project(shot.pos))
        if out.outcome == "none":
            continue
        margins.append(out.margin_score)
        labels.append(1.0 if shot.outcome == "make" else 0.0)
    return np.array(margins), np.array(labels)


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 20260713)
    az, h = cfg.get("azimuth_deg", 55), cfg.get("height_m", 1.5)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=9.0)
    ell = compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse
    # val-cal (fit) and test (report) are DIFFERENT sessions (leakage discipline).
    cal = generate_session(venue_scene("gym_A", azimuth_deg=az, height_m=h), n_shots=120, fps=60, seed=seed)
    test = generate_session(venue_scene("gym_B", azimuth_deg=az, height_m=h), n_shots=120, fps=60, seed=seed + 99)
    m_cal, y_cal = _collect(cam, ell, cal)
    m_test, y_test = _collect(cam, ell, test)

    temp = TemperatureScaler().fit(m_cal, y_cal)
    platt = PlattScaler().fit(m_cal, y_cal)
    sig = lambda x: 1 / (1 + np.exp(-x))
    prob_sets = {"uncalibrated": sig(m_test), "temperature": temp.predict(m_test), "platt": platt.predict(m_test)}

    rows, metrics = [], {}
    for name, probs in prob_sets.items():
        ece = expected_calibration_error(probs, y_test, n_bins=10)
        brier = brier_score(probs, y_test)
        rows.append({"method": name, "ece": round(ece, 4), "brier": round(brier, 4),
                     "n_test": len(y_test)})
        metrics[f"ece_{name}"] = ece
        metrics[f"brier_{name}"] = brier
    metrics["temperature_T"] = temp.T

    fig = reliability_diagram(prob_sets, y_test, n_bins=10, title="A9 — reliability on test (regime: S)")
    fig_path = save_fig(fig.figure, "a9_reliability")
    run_id = log_run("bball-A9", "a9_calibration", params={"seed": seed, "temperature_T": round(temp.T, 3)},
                     metrics=metrics, figures={"a9": fig_path}, summary_rows=rows)
    print(f"[A9] run_id={run_id} ECE raw={metrics['ece_uncalibrated']:.3f} temp={metrics['ece_temperature']:.3f} (T={temp.T:.2f})")
    return {"run_id": run_id, "rows": rows}
