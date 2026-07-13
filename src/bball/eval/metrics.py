"""Evaluation metrics with honest denominators (plan §1.2, §8).

Event P/R/F1 with a +-0.25 s temporal tolerance and false-attempts-per-hour on shot-free
footage (T1); T3 court-position error in cm; zone confusion; track completeness split by
occlusion; per-axis T5 accuracy (left/right and short/long reported SEPARATELY, since a
single camera collapses one depth axis and an aggregate would hide it). Calibration metrics
(ECE/Brier) live in events.calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def event_prf(pred_times: list[float], gt_times: list[float], *, tol_s: float = 0.25) -> PRF:
    """Greedy one-to-one temporal matching within tolerance. Each GT event matches at most
    one prediction and vice versa (nearest first)."""
    pred = sorted(pred_times)
    gt = sorted(gt_times)
    used_pred = set()
    tp = 0
    for g in gt:
        best, best_d = -1, tol_s + 1e-9
        for j, p in enumerate(pred):
            if j in used_pred:
                continue
            d = abs(p - g)
            if d <= best_d:
                best, best_d = j, d
        if best >= 0:
            used_pred.add(best)
            tp += 1
    fp = len(pred) - len(used_pred)
    fn = len(gt) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return PRF(precision, recall, f1, tp, fp, fn)


def outcome_prf(pred_labels: list[str], gt_labels: list[str], positive: str = "make") -> PRF:
    """P/R/F1 for a binary outcome (make vs miss) over matched events."""
    tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == positive and g == positive)
    fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == positive and g != positive)
    fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != positive and g == positive)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return PRF(precision, recall, f1, tp, fp, fn)


def false_attempts_per_hour(n_false_attempts: int, duration_s: float) -> float:
    if duration_s <= 0:
        return float("nan")
    return n_false_attempts / (duration_s / 3600.0)


def t3_error_cm(pred_xy: np.ndarray, gt_xy: np.ndarray) -> dict:
    """Court-position error in cm: median and P90."""
    pred = np.atleast_2d(np.asarray(pred_xy, float))
    gt = np.atleast_2d(np.asarray(gt_xy, float))
    err_m = np.sqrt(((pred - gt) ** 2).sum(axis=1))
    err_cm = err_m * 100.0
    return {"median_cm": float(np.median(err_cm)), "p90_cm": float(np.percentile(err_cm, 90)),
            "mean_cm": float(err_cm.mean()), "n": int(len(err_cm))}


def zone_confusion(pred_zones: list[str], gt_zones: list[str], zones=("short-range", "midrange", "3PT")) -> dict:
    idx = {z: i for i, z in enumerate(zones)}
    mat = np.zeros((len(zones), len(zones)), int)
    for p, g in zip(pred_zones, gt_zones):
        if p in idx and g in idx:
            mat[idx[g], idx[p]] += 1
    total = mat.sum()
    acc = float(np.trace(mat) / total) if total else float("nan")
    return {"matrix": mat, "zones": list(zones), "accuracy": acc}


def track_completeness(observed: np.ndarray, filled: np.ndarray | None = None,
                       occlusion: np.ndarray | None = None) -> dict:
    """% of flight frames with a position estimate. `observed` = real detections; `filled`
    (optional) = observed OR bridged. Reports both, and a split by occlusion state."""
    observed = np.asarray(observed, bool)
    out = {"observed_pct": float(observed.mean())}
    if filled is not None:
        out["filled_pct"] = float(np.asarray(filled, bool).mean())
    if occlusion is not None:
        occ = np.asarray(occlusion, float)
        for name, mask in [("clean", occ < 0.2), ("occluded", occ >= 0.2)]:
            if mask.any():
                out[f"observed_pct_{name}"] = float(observed[mask].mean())
    return out


def per_axis_t5_accuracy(preds: list[dict], gts: list[dict]) -> dict:
    """Left/right and short/long accuracy reported SEPARATELY. Each pred/gt is a dict with
    'left_right' and 'short_long' string labels (or '' / None). Only axes that are shown
    (non-empty) count toward that axis' denominator."""
    def axis_acc(key):
        n = c = 0
        for p, g in zip(preds, gts):
            gl = g.get(key)
            pl = p.get(key)
            if not gl:
                continue
            n += 1
            c += int(pl == gl)
        return {"accuracy": (c / n) if n else float("nan"), "n": n}

    return {"left_right": axis_acc("left_right"), "short_long": axis_acc("short_long")}
