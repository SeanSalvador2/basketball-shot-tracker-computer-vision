"""A4 — player tracker comparison under occlusion (plan §7, regime S).

Three arms, each isolating one mechanism:
  1. greedy-IoU              — last-seen box, greedy IoU match, high-score dets only (no motion
                               model, no recovery)                          [baseline]
  2. Kalman + IoU            — CV-Kalman prediction + Hungarian IoU, high-score dets only
                               (adds the motion model)
  3. Kalman + IoU + recovery — full ByteTrack: adds the low-score second association pass
                               (adds occlusion recovery)

Scenes come from the existing multi-agent simulator (`synth.scenarios.simulate_players`):
players walk random waypoints, are projected to image boxes (feet+head), and given an
occlusion-aware detection model — a player behind a nearer one loses detection score
(dropping into the low-score band, or missing entirely under heavy overlap). That band is
exactly what arm 3's second pass is built to recover, so the arms should separate on
ID switches / IDF1 / HOTA precisely under occlusion.

Metrics (`bball.eval.metrics`): HOTA (single-alpha, detection-matched simplification — see the
metric's docstring), IDF1, ID switches. Regime S.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bball.ablations.common import log_run, save_fig
from bball.detect.interfaces import Detection, iou
from bball.eval.metrics import hota_simplified, id_switches, idf1
from bball.lift.court_model import get_court
from bball.synth.camera import make_camera
from bball.synth.scenarios import simulate_players
from bball.track.association import ByteTrackConfig, ByteTrackPlayerTracker


# --------------------------------------------------------------------------- #
# Scene: project players to boxes, apply occlusion-aware detection noise
# --------------------------------------------------------------------------- #
def _player_box(cam, xy, height_m):
    feet = cam.project(np.array([[xy[0], xy[1], 0.0]]))[0]
    head = cam.project(np.array([[xy[0], xy[1], height_m]]))[0]
    if np.isnan(feet).any() or np.isnan(head).any():
        return None
    top, bottom = float(head[1]), float(feet[1])
    if bottom - top < 6:
        return None
    w = 0.35 * (bottom - top)
    cx = float(feet[0])
    return (cx - w / 2, top, cx + w / 2, bottom)


def _depth(cam, xy):
    p = np.array([xy[0], xy[1], 0.0])
    return float((cam.R @ (p - cam.position))[2])


def build_scene(seed: int, n_players: int, n_frames: int, cam, court, fps=30.0,
                spread: float = 0.45, occ_drop: float = 0.55, occ_k: float = 1.0):
    """Return (gt_frames, det_frames). gt_frames[t] = {pid: box}; det_frames[t] = [Detection].

    Occlusion model: a player behind a nearer overlapping one loses detection score in
    proportion to the overlap (dropping into the low-score band), and is dropped entirely
    once overlap exceeds `occ_drop` — a multi-frame miss that must be coasted and recovered.
    `spread` < 1 compresses the walkers toward court centre so they overlap heavily in image
    space (the existing simulator's default walk is too dispersed to occlude)."""
    players = simulate_players(n_players, n_frames, court, fps=fps, seed=seed)
    center = np.array([0.0, 4.0])
    for pl in players:  # cluster toward centre to force image-space overlap
        pl["pos_xy"] = center + (pl["pos_xy"] - center) * spread
    rng = np.random.default_rng(seed + 7)
    gt_frames, det_frames = [], []
    for t in range(n_frames):
        boxes, depths, pids = {}, {}, []
        for pl in players:
            xy = pl["pos_xy"][t]
            box = _player_box(cam, xy, pl["height_m"])
            if box is None:
                continue
            # keep boxes with any overlap of the image rectangle
            if box[2] < 0 or box[0] > cam.width_px or box[3] < 0 or box[1] > cam.height_px:
                continue
            pid = pl["player_id"]
            boxes[pid] = box
            depths[pid] = _depth(cam, xy)
            pids.append(pid)
        gt_frames.append(dict(boxes))
        # occlusion-aware detections
        dets = []
        for pid in pids:
            occ = 0.0
            for other in pids:
                if other == pid:
                    continue
                if depths[other] < depths[pid]:  # other is nearer -> occludes pid
                    occ = max(occ, iou(boxes[pid], boxes[other]))
            base = 0.9
            score = base * (1.0 - occ_k * occ) + rng.normal(0, 0.03)
            score = float(np.clip(score, 0.0, 0.99))
            if occ > occ_drop or score < 0.1:
                continue  # heavily occluded -> no detection (a multi-frame gap to coast/recover)
            b = boxes[pid]
            j = rng.normal(0, 3.0, 4)
            dets.append(Detection(bbox=(b[0] + j[0], b[1] + j[1], b[2] + j[2], b[3] + j[3]),
                                  score=score, label="person", frame_idx=t))
        det_frames.append(dets)
    return gt_frames, det_frames


# --------------------------------------------------------------------------- #
# Arm 1 — greedy IoU, no motion model, high-score only
# --------------------------------------------------------------------------- #
class GreedyIoUTracker:
    def __init__(self, iou_thr=0.2, high_thresh=0.5, max_age=30, min_hits=2):
        self.iou_thr, self.high, self.max_age, self.min_hits = iou_thr, high_thresh, max_age, min_hits
        self.tracks = {}  # id -> dict(box, tsu, hits)
        self._next = 0

    def update(self, detections):
        dets = [d for d in detections if d.score >= self.high]
        tids = list(self.tracks)
        used_d = set()
        # greedy: repeatedly take the highest-IoU (track, det) pair
        pairs = []
        for ti in tids:
            for di, d in enumerate(dets):
                v = iou(self.tracks[ti]["box"], d.bbox)
                if v >= self.iou_thr:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        matched_t = set()
        for v, ti, di in pairs:
            if ti in matched_t or di in used_d:
                continue
            self.tracks[ti]["box"] = dets[di].bbox
            self.tracks[ti]["tsu"] = 0
            self.tracks[ti]["hits"] += 1
            matched_t.add(ti)
            used_d.add(di)
        for ti in tids:
            if ti not in matched_t:
                self.tracks[ti]["tsu"] += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks[self._next] = {"box": d.bbox, "tsu": 0, "hits": 1}
                self._next += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["tsu"] <= self.max_age}
        return {k: v["box"] for k, v in self.tracks.items()
                if v["hits"] >= self.min_hits or v["tsu"] == 0}


def _run_greedy(det_frames):
    tr = GreedyIoUTracker()
    return [tr.update(dets) for dets in det_frames]


def _run_bytetrack(det_frames, recovery: bool, process_std=25.0, meas_std=5.0, max_age=30):
    # recovery off  => low_thresh == high_thresh so the second pass never fires. Arms 2 and 3
    # share every other setting, so the only difference measured is the low-score pass.
    cfg = ByteTrackConfig(high_thresh=0.5, low_thresh=(0.1 if recovery else 0.5), iou_match=0.2,
                          process_std=process_std, meas_std=meas_std, max_age=max_age)
    tr = ByteTrackPlayerTracker(cfg)
    return [tr.update(dets) for dets in det_frames]


def _arms(process_std, meas_std, max_age):
    return {
        "greedy-IoU": _run_greedy,
        "Kalman+IoU": lambda df: _run_bytetrack(df, False, process_std, meas_std, max_age),
        "Kalman+IoU+recovery": lambda df: _run_bytetrack(df, True, process_std, meas_std, max_age),
    }


def _plot(rows):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    arms = [r["arm"] for r in rows]
    for a, key, ylab in zip(ax, ["hota", "idf1", "id_switches"],
                            ["HOTA (simplified)", "IDF1", "ID switches (lower=better)"]):
        vals = [r[key] for r in rows]
        color = "tab:red" if key == "id_switches" else "tab:blue"
        a.bar(arms, vals, color=color)
        a.set_ylabel(ylab)
        a.tick_params(axis="x", rotation=20, labelsize=8)
        if key != "id_switches":
            a.set_ylim(0, 1.02)
    fig.suptitle("A4 — player tracker arms under occlusion (regime S; HOTA simplified, single-alpha)")
    fig.tight_layout()
    return fig


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 20260713)
    n_players = cfg.get("n_players", 6)
    n_frames = cfg.get("n_frames", 150)
    n_scenes = cfg.get("n_scenes", 4)
    az, h = cfg.get("azimuth_deg", 25), cfg.get("height_m", 2.2)
    cam = make_camera(azimuth_deg=az, height_m=h, distance_m=cfg.get("distance_m", 11.0))
    court = get_court(cfg.get("court_spec", "nba"))
    spread = cfg.get("spread", 0.45)
    occ_drop = cfg.get("occ_drop", 0.55)
    process_std = cfg.get("kf_process_std", 25.0)
    meas_std = cfg.get("kf_meas_std", 5.0)
    max_age = cfg.get("kf_max_age", 30)

    scenes = [build_scene(seed + s, n_players, n_frames, cam, court, spread=spread,
                          occ_drop=occ_drop) for s in range(n_scenes)]
    # occlusion diagnostics: mean fraction of GT boxes with no detection this frame
    tot_gt = sum(len(f) for gt, _ in scenes for f in gt)
    tot_det = sum(len(d) for _, dd in scenes for d in dd)
    miss_rate = 1.0 - tot_det / max(tot_gt, 1)

    rows = []
    for arm, fn in _arms(process_std, meas_std, max_age).items():
        H = I = S = 0.0
        for gt_frames, det_frames in scenes:
            pred = fn(det_frames)
            H += hota_simplified(gt_frames, pred)["hota"]
            I += idf1(gt_frames, pred)["idf1"]
            S += id_switches(gt_frames, pred)
        rows.append({"arm": arm, "hota": round(H / n_scenes, 3), "idf1": round(I / n_scenes, 3),
                     "id_switches": round(S / n_scenes, 2), "regime": "S"})

    fig = _plot(rows)
    fig_path = save_fig(fig, "a4_tracker")
    metrics = {}
    for r in rows:
        tag = r["arm"].replace("+", "_").replace("-", "")
        metrics[f"hota_{tag}"] = r["hota"]
        metrics[f"idf1_{tag}"] = r["idf1"]
        metrics[f"idsw_{tag}"] = r["id_switches"]
    metrics["occlusion_miss_rate"] = round(miss_rate, 3)
    run_id = log_run("bball-A4", "a4_tracker",
                     params={"seed": seed, "n_players": n_players, "n_frames": n_frames,
                             "n_scenes": n_scenes, "azimuth_deg": az, "regime": "S",
                             "hota": "simplified single-alpha (see metric docstring)"},
                     metrics=metrics, figures={"a4": fig_path}, summary_rows=rows)
    print(f"[A4] run_id={run_id} occ_miss_rate={miss_rate:.3f}")
    for r in rows:
        print("   ", r)
    return {"run_id": run_id, "rows": rows}
