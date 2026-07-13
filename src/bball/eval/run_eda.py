"""EDA tooling (plan §3) — one command runs every analysis on the synthetic sessions and
writes figures to reports/figures/eda/. Each analysis de-risks a specific design decision;
Stage B re-runs the same commands on real footage and the sim-vs-real comparison audits the
synthetic engine (gate G4).

    python -m bball.eval.run_eda --config configs/eda.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bball.synth.camera import apparent_ball_radius_px, make_camera, project_trajectory  # noqa: E402
from bball.synth.physics import RELEASE_ANGLE_RANGE, RELEASE_SPEED_RANGE, RIM_HEIGHT_M  # noqa: E402
from bball.synth.render import compute_rim_image_geometry, occlusion_fraction  # noqa: E402
from bball.synth.scenarios import generate_session, venue_scene  # noqa: E402
from bball.utils.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EDA_DIR = REPO / "reports" / "figures" / "eda"


def _placements(cfg):
    return cfg.get("placements", [
        {"venue": "gym_A", "azimuth_deg": 45, "height_m": 3.0},
        {"venue": "gym_B", "azimuth_deg": 30, "height_m": 1.5},
        {"venue": "outdoor_A", "azimuth_deg": 60, "height_m": 3.0},
        {"venue": "outdoor_B", "azimuth_deg": 90, "height_m": 2.5},
    ])


def _sessions(cfg):
    out = []
    for i, p in enumerate(_placements(cfg)):
        scene = venue_scene(p["venue"], azimuth_deg=p["azimuth_deg"], height_m=p["height_m"])
        cam = make_camera(azimuth_deg=p["azimuth_deg"], height_m=p["height_m"], distance_m=9.0)
        sess = generate_session(scene, n_shots=cfg.get("n_shots", 60), fps=60, seed=cfg.get("seed", 7) + i)
        out.append({"placement": p, "scene": scene, "cam": cam, "session": sess})
    return out


def ball_size(sessions) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4))
    findings = {}
    for s in sessions:
        radii = []
        for shot in s["session"].shots:
            r = apparent_ball_radius_px(s["cam"], shot.pos)
            radii.extend(r[np.isfinite(r)].tolist())
        radii = np.array(radii)
        diam = radii * 2
        ax.hist(diam, bins=40, alpha=0.5, label=f"{s['placement']['venue']} az{s['placement']['azimuth_deg']}")
        findings[s["placement"]["venue"]] = {"median_diam_px": float(np.median(diam)),
                                             "p10_px": float(np.percentile(diam, 10)),
                                             "p90_px": float(np.percentile(diam, 90))}
    ax.axvspan(20, 40, color="green", alpha=0.1, label="Phase-0 assumed 20-40 px")
    ax.set_xlabel("ball apparent diameter (px)")
    ax.set_ylabel("frames")
    ax.set_title("EDA — ball apparent size vs camera placement")
    ax.legend(fontsize=7)
    _save(fig, "eda_ball_size")
    return {"analysis": "ball_size", "figure": "eda_ball_size.png", "per_placement": findings,
            "consumer": "detector input resolution (A3), heatmap-vs-bbox framing"}


def motion_blur(sessions) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4))
    findings = {}
    for fps_label, decim in [("240fps", 1), ("60fps", 4), ("30fps", 8)]:
        streaks = []
        for s in sessions:
            for shot in s["session"].shots:
                img = project_trajectory(s["cam"], shot.pos)[::decim]
                d = np.linalg.norm(np.diff(img, axis=0), axis=1)
                streaks.extend(d[np.isfinite(d)].tolist())
        streaks = np.array(streaks)
        ax.hist(streaks, bins=50, histtype="step", label=f"{fps_label} (median {np.median(streaks):.0f}px)")
        findings[fps_label] = {"median_streak_px": float(np.median(streaks)), "p90_px": float(np.percentile(streaks, 90))}
    ax.set_xlabel("per-frame ball displacement / streak length (px)")
    ax.set_ylabel("frames")
    ax.set_xlim(0, 120)
    ax.set_title("EDA — motion blur (streak length) vs frame rate")
    ax.legend(fontsize=8)
    _save(fig, "eda_motion_blur")
    return {"analysis": "motion_blur", "figure": "eda_motion_blur.png", "by_fps": findings,
            "consumer": "fps recommendation, blur augmentation realism"}


def occlusion_timeline(sessions) -> dict:
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for s in sessions:
        cam = s["cam"]
        rim_geom = compute_rim_image_geometry(cam, (0.0, 0.0))
        occ_fracs = []
        for shot in s["session"].shots:
            img = project_trajectory(cam, shot.pos)
            rad = apparent_ball_radius_px(cam, shot.pos)
            occ = [occlusion_fraction(None if np.isnan(img[i]).any() else img[i],
                                      float(rad[i]) if not np.isnan(rad[i]) else 3.0, rim_geom)
                   for i in range(shot.n_frames)]
            occ_fracs.append(np.mean(np.array(occ) > 0.5))
        rows.append((s["placement"]["azimuth_deg"], float(np.mean(occ_fracs))))
    rows.sort()
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-")
    ax.set_xlabel("camera azimuth (deg)")
    ax.set_ylabel("mean fraction of flight frames occluded (>0.5)")
    ax.set_title("EDA — occlusion vs camera azimuth")
    ax.grid(alpha=0.3)
    _save(fig, "eda_occlusion")
    return {"analysis": "occlusion", "figure": "eda_occlusion.png", "by_azimuth": rows,
            "consumer": "bridging design (A5), collection guidance"}


def rim_geometry(sessions) -> dict:
    """Surfaces the finding that a camera near rim height views the rim near edge-on."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for s in sessions:
        for h in [1.5, 2.5, 3.0, 3.5, 4.5]:
            cam = make_camera(azimuth_deg=s["placement"]["azimuth_deg"], height_m=h, distance_m=9.0)
            ell = compute_rim_image_geometry(cam, (0.0, 0.0)).ellipse
            rows.append({"azimuth": s["placement"]["azimuth_deg"], "height_m": h,
                         "a_px": round(ell.a, 1), "b_px": round(ell.b, 1),
                         "b_over_a": round(ell.b / max(ell.a, 1e-6), 3)})
        break  # one azimuth is enough to show the height dependence
    hs = [r["height_m"] for r in rows]
    ax.plot(hs, [r["b_over_a"] for r in rows], "o-")
    ax.axvline(RIM_HEIGHT_M, color="red", ls="--", label=f"rim height {RIM_HEIGHT_M} m (edge-on)")
    ax.set_xlabel("camera height (m)")
    ax.set_ylabel("rim ellipse minor/major axis ratio (roundness)")
    ax.set_title("EDA — rim ellipse roundness vs camera height")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "eda_rim_geometry")
    return {"analysis": "rim_geometry", "figure": "eda_rim_geometry.png", "rows": rows,
            "consumer": "rim-normalized FSM viability; camera-placement guidance (avoid ~rim height)"}


def trajectory_stats(sessions) -> dict:
    angles, speeds, apexes, flights = [], [], [], []
    for s in sessions:
        for shot in s["session"].shots:
            angles.append(shot.release_angle_deg)
            speeds.append(shot.release_speed)
            apexes.append(shot.apex_height_m)
            flights.append(shot.events["rim_arrival_t"] - shot.events["release_t"])
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, data, name, cite in [
        (axes[0, 0], angles, "release angle (deg)", RELEASE_ANGLE_RANGE),
        (axes[0, 1], speeds, "release speed (m/s)", RELEASE_SPEED_RANGE),
        (axes[1, 0], apexes, "apex height (m)", None),
        (axes[1, 1], flights, "flight time (s)", None),
    ]:
        ax.hist(data, bins=30, color="tab:blue", alpha=0.8)
        if cite:
            ax.axvspan(cite[0], cite[1], color="green", alpha=0.15, label="cited range")
            ax.legend(fontsize=7)
        ax.set_xlabel(name)
    fig.suptitle("EDA — trajectory statistics (grounds the synthetic engine, gate G4)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "eda_trajectory_stats")
    return {"analysis": "trajectory_stats", "figure": "eda_trajectory_stats.png",
            "median_apex_m": float(np.median(apexes)), "median_flight_s": float(np.median(flights)),
            "consumer": "synthetic engine parameter grounding; shot-attempt thresholds"}


def class_balance(sessions) -> dict:
    from collections import Counter

    outcomes = Counter()
    zones = Counter()
    for s in sessions:
        for shot in s["session"].shots:
            outcomes[shot.outcome] += 1
            zones[shot.meta.get("zone", "?")] += 1
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(list(outcomes.keys()), list(outcomes.values()), color=["tab:green", "tab:red"])
    axes[0].set_title("make/miss balance")
    axes[1].bar(list(zones.keys()), list(zones.values()), color="tab:blue")
    axes[1].set_title("zone balance")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "eda_class_balance")
    return {"analysis": "class_balance", "figure": "eda_class_balance.png",
            "outcomes": dict(outcomes), "zones": dict(zones),
            "consumer": "split stratification; loss weighting"}


def _save(fig, name):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_all(cfg: dict) -> dict:
    sessions = _sessions(cfg)
    findings = {
        "ball_size": ball_size(sessions),
        "motion_blur": motion_blur(sessions),
        "occlusion": occlusion_timeline(sessions),
        "rim_geometry": rim_geometry(sessions),
        "trajectory_stats": trajectory_stats(sessions),
        "class_balance": class_balance(sessions),
    }
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDA_DIR / "eda_findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"[EDA] wrote {len(findings)} analyses to {EDA_DIR}")
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run EDA analyses")
    ap.add_argument("--config", default=str(REPO / "configs" / "eda.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    run_all(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
