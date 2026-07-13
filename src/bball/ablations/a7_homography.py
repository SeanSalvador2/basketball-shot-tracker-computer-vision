"""A7 — homography: 4 vs 6 vs 8 points x click noise {1,2,5,10}px x refinement ON/OFF.

Hypothesis: 6+ points with RANSAC+LM roughly halve P90 error vs raw 4-point DLT at realistic
(2-5 px) click noise; zone accuracy is insensitive except in the on-line band. Deliverable:
cm-error-vs-noise curves + a court error-isoline map (the camera-placement guide). Regime: S
(Monte Carlo); confirmed on marked spots in Stage B.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.lift.court_model import classify_zone, get_court, landmark_points
from bball.lift.homography import apply_homography, dlt_homography, estimate_homography
from bball.lift.projection import Camera


def _test_grid(court, n=14):
    xs = np.linspace(-court.sideline_x_m + 0.5, court.sideline_x_m - 0.5, n)
    ys = np.linspace(0.3, court.three_arc_radius_m + 1.0, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 20260713)
    n_points_list = cfg.get("n_points", [4, 6, 8])
    noises = cfg.get("click_noise_px", [1, 2, 5, 10])
    n_mc = cfg.get("n_montecarlo", 60)
    cam = Camera.from_look_at(azimuth_deg=cfg.get("azimuth_deg", 45), height_m=cfg.get("height_m", 3.0),
                              distance_m=cfg.get("distance_m", 9.0), target_height_m=0.0)
    court = get_court(cfg.get("court_spec", "nba"))
    lms = np.array(list(landmark_points(court).values()))
    lms_img_true = cam.project(np.hstack([lms, np.zeros((len(lms), 1))]))
    grid = _test_grid(court)
    grid_img_true = cam.project(np.hstack([grid, np.zeros((len(grid), 1))]))
    gt_zones = [classify_zone(court, x, y) for (x, y) in grid]

    rows = []
    curves = {}  # (n_points, refine) -> {noise: p50, ...}
    for npts in n_points_list:
        for refine in [False, True]:
            curves[(npts, refine)] = {}
            for sigma in noises:
                med_errs, p90_errs, zone_accs = [], [], []
                for mc in range(n_mc):
                    rng = np.random.default_rng(seed + mc + npts * 100 + int(refine) * 50 + sigma)
                    src = lms[:npts]
                    dst = lms_img_true[:npts] + rng.normal(0, sigma, size=(npts, 2))
                    # Refinement = LM over all correspondences (minimizes geometric reprojection
                    # error). RANSAC is for GROSS click outliers, not the gaussian click noise
                    # swept here — applying it with a tight gate would wrongly discard valid
                    # noisy points, so the honest DLT-vs-DLT+LM comparison uses LM.
                    if refine:
                        H = estimate_homography(src, dst, use_ransac=False, refine=True).H
                    else:
                        H = dlt_homography(src, dst)
                    Hinv = np.linalg.inv(H)
                    est = apply_homography(Hinv, grid_img_true)
                    err_cm = np.sqrt(((est - grid) ** 2).sum(axis=1)) * 100
                    med_errs.append(np.median(err_cm))
                    p90_errs.append(np.percentile(err_cm, 90))
                    est_zones = [classify_zone(court, x, y) for (x, y) in est]
                    zone_accs.append(np.mean([a == b for a, b in zip(est_zones, gt_zones)]))
                p50 = float(np.mean(med_errs))
                p90 = float(np.mean(p90_errs))
                za = float(np.mean(zone_accs))
                curves[(npts, refine)][sigma] = p50
                rows.append({"n_points": npts, "refine": refine, "click_noise_px": sigma,
                             "median_cm": round(p50, 1), "p90_cm": round(p90, 1), "zone_acc": round(za, 3)})

    fig = _plot_curves(curves, noises)
    fig_path = save_fig(fig, "a7_homography_error")
    iso_fig = _plot_isolines(cam, court, grid, seed)
    iso_path = save_fig(iso_fig, "a7_error_isolines")
    metrics = {f"med_cm_p{r['n_points']}_ref{int(r['refine'])}_n{r['click_noise_px']}": r["median_cm"] for r in rows}
    run_id = log_run("bball-A7", "a7_homography", params={"n_points": n_points_list, "noises": noises,
                     "n_montecarlo": n_mc, "seed": seed}, metrics=metrics,
                     figures={"curves": fig_path, "isolines": iso_path}, summary_rows=rows)
    print(f"[A7] run_id={run_id} figures={fig_path.name},{iso_path.name}")
    return {"run_id": run_id, "rows": rows}


def _plot_curves(curves, noises):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (npts, refine), d in sorted(curves.items()):
        ys = [d[s] for s in noises]
        style = "-o" if refine else "--s"
        ax.plot(noises, ys, style, label=f"{npts} pts {'+RANSAC+LM' if refine else 'DLT only'}")
    ax.set_xlabel("click noise (px)")
    ax.set_ylabel("median court-position error (cm)")
    ax.set_title("A7 — homography error vs click noise (regime: S)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def _plot_isolines(cam, court, grid, seed, sigma=3.0):
    rng = np.random.default_rng(seed)
    lms = np.array(list(landmark_points(court).values()))
    lms_img = cam.project(np.hstack([lms, np.zeros((len(lms), 1))]))
    errs = np.zeros(len(grid))
    grid_img = cam.project(np.hstack([grid, np.zeros((len(grid), 1))]))
    for mc in range(40):
        dst = lms_img + rng.normal(0, sigma, size=lms_img.shape)
        H = estimate_homography(lms, dst, use_ransac=True, refine=True, seed=mc).H
        est = apply_homography(np.linalg.inv(H), grid_img)
        errs += np.sqrt(((est - grid) ** 2).sum(axis=1)) * 100
    errs /= 40
    n = int(np.sqrt(len(grid)))
    fig, ax = plt.subplots(figsize=(6, 6))
    from bball.viz.court import plot_halfcourt

    plot_halfcourt(court, ax=ax)
    cs = ax.tricontourf(grid[:, 0], grid[:, 1], errs, levels=10, cmap="YlOrRd", alpha=0.6)
    ax.tricontour(grid[:, 0], grid[:, 1], errs, levels=[10, 20, 40], colors="k", linewidths=0.6)
    fig.colorbar(cs, ax=ax, label="mean T3 error (cm)")
    ax.set_title(f"A7 — court error isolines @ {sigma}px click noise (regime: S)")
    return fig
