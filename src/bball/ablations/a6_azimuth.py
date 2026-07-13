"""A6 — camera azimuth sweep x height -> per-axis T5 (miss-direction) accuracy.

Hypothesis (stated before results): left/right accuracy is ~flat across azimuth (it is
image-plane geometry near the rim); short/long decays toward end-on views (the depth axis
collapses into the optical axis); 45-60 deg elevated is the knee. THIS CURVE IS THE
PRODUCT'S CAMERA-PLACEMENT GUIDANCE. Regime: S (synthetic — this sweep is exactly what
synthetic is for).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.events.miss_direction import decompose_miss
from bball.lift.rim_frame import rim_3d_center
from bball.synth.camera import apparent_ball_radius_px, make_camera
from bball.synth.physics import generate_shot
from bball.track.ballistic import reconstruct_flight_3d

PURE_DIRS = ["left", "right", "short", "long"]


def _one_miss_axis_calls(cam, shooter_xy, direction, mag, seed, jitter_px=2.0, feet_noise_m=0.25):
    shot = generate_shot(release_xy=shooter_xy, hoop_ground_xy=(0.0, 0.0), outcome="miss",
                         miss_direction=direction, miss_magnitude_m=mag, seed=seed)
    mask = (shot.t >= shot.events["release_t"]) & (shot.t <= shot.events["rim_arrival_t"] + 0.05)
    pos = shot.pos[mask]
    t = shot.t[mask]
    if len(pos) < 6:
        return None
    rng = np.random.default_rng(seed)
    # Realistic error sources. At end-on views the depth axis is ill-conditioned, so the fit
    # leans on the release-point anchor and the ball-size cue — both uncertain in the field
    # (feet position carries T3-level error; a +-1 px diameter is a few-% depth error). Those
    # uncertainties are what let short/long degrade toward chance end-on while reprojection
    # keeps it accurate side-on. This is the honest content of A6.
    img = cam.project(pos) + rng.normal(0, jitter_px, size=(len(pos), 2))
    rad = apparent_ball_radius_px(cam, pos) * (1.0 + rng.normal(0, 0.15, size=len(pos)))
    feet_est = shot.release_xy + rng.normal(0, feet_noise_m, size=2)
    traj = reconstruct_flight_3d(img, t, cam, shooter_feet_xy=feet_est,
                                 rim_center_3d=rim_3d_center((0.0, 0.0)), ball_radius_px=rad)
    res = decompose_miss(traj, rim_3d_center((0.0, 0.0)), shot.release_xy, dead_zone_m=0.0)
    return res, traj.confidence


def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg.get("seed", 20260713))
    azimuths = cfg.get("azimuths", [15, 30, 45, 60, 75, 90])
    heights = cfg.get("heights", [1.5, 3.0])
    n_per = cfg.get("n_per_dir", 25)
    distance = cfg.get("distance_m", 9.0)

    results = {}   # (height, azimuth) -> {'left_right': acc, 'short_long': acc, 'conf': mean}
    rows = []
    for h in heights:
        for az in azimuths:
            cam = make_camera(azimuth_deg=az, height_m=h, distance_m=distance)
            lr_ok = lr_n = sl_ok = sl_n = 0
            confs = []
            for d in PURE_DIRS:
                for _ in range(n_per):
                    # Central shooters keep the depth axis ~along +y so azimuth sweeps the
                    # camera consistently relative to it (clean crossing curves).
                    sx = np.array([rng.uniform(-2.0, 2.0), rng.uniform(5.5, 7.2)])
                    out = _one_miss_axis_calls(cam, sx, d, float(rng.uniform(0.4, 0.9)),
                                               int(rng.integers(1 << 31)))
                    if out is None:
                        continue
                    res, conf = out
                    confs.append(conf)
                    if d in ("left", "right"):
                        lr_n += 1
                        lr_ok += int(res.left_right.label == d)
                    else:
                        sl_n += 1
                        sl_ok += int(res.short_long.label == d)
            lr_acc = lr_ok / max(lr_n, 1)
            sl_acc = sl_ok / max(sl_n, 1)
            results[(h, az)] = {"left_right": lr_acc, "short_long": sl_acc, "conf": float(np.mean(confs))}
            rows.append({"height_m": h, "azimuth_deg": az, "left_right_acc": round(lr_acc, 3),
                         "short_long_acc": round(sl_acc, 3), "mean_confidence": round(float(np.mean(confs)), 3),
                         "n_lr": lr_n, "n_sl": sl_n})

    fig = _plot(results, azimuths, heights)
    fig_path = save_fig(fig, "a6_azimuth_sweep")
    metrics = {}
    for (h, az), r in results.items():
        metrics[f"lr_acc_h{h}_az{az}"] = r["left_right"]
        metrics[f"sl_acc_h{h}_az{az}"] = r["short_long"]
    run_id = log_run("bball-A6", "a6_azimuth_sweep", params={"azimuths": azimuths, "heights": heights,
                     "n_per_dir": n_per, "seed": cfg.get("seed", 20260713)},
                     metrics=metrics, figures={"a6": fig_path}, summary_rows=rows)
    print(f"[A6] run_id={run_id} figure={fig_path}")
    return {"run_id": run_id, "rows": rows}


def _plot(results, azimuths, heights):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for h in heights:
        lr = [results[(h, az)]["left_right"] for az in azimuths]
        sl = [results[(h, az)]["short_long"] for az in azimuths]
        axes[0].plot(azimuths, lr, "o-", label=f"h={h} m")
        axes[1].plot(azimuths, sl, "s-", label=f"h={h} m")
    axes[0].set_title("Left/right accuracy (image-plane, robust)")
    axes[1].set_title("Short/long accuracy (depth, azimuth-dependent)")
    for ax in axes:
        ax.set_xlabel("camera azimuth (deg from baseline)")
        ax.set_ylabel("per-axis accuracy")
        ax.set_ylim(0.3, 1.02)
        ax.axhline(0.5, color="0.7", ls=":", lw=1)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("A6 — miss-direction accuracy vs camera placement (regime: S)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
