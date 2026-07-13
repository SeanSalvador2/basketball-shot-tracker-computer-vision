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


def detection_recall_at_iou(pred_boxes: list[list[tuple]], gt_boxes: list[list[tuple]],
                            iou_thr: float = 0.3) -> dict:
    """Single-class box recall at an IoU threshold, greedy one-to-one per frame.

    `pred_boxes` / `gt_boxes` are per-frame lists of (x0,y0,x1,y1). A GT box counts as
    recalled if some as-yet-unused prediction in its frame overlaps it at IoU >= iou_thr.
    Used by A3 (detector resolution ablation, regime S)."""
    from bball.detect.interfaces import iou as _iou

    tp = n_gt = 0
    for preds, gts in zip(pred_boxes, gt_boxes):
        used = set()
        for g in gts:
            n_gt += 1
            best_j, best_i = -1, iou_thr
            for j, p in enumerate(preds):
                if j in used:
                    continue
                v = _iou(p, g)
                if v >= best_i:
                    best_j, best_i = j, v
            if best_j >= 0:
                used.add(best_j)
                tp += 1
    return {"recall": tp / max(n_gt, 1), "tp": tp, "n_gt": n_gt}


def average_precision_at_iou(pred_boxes: list[list[tuple]], pred_scores: list[list[float]],
                             gt_boxes: list[list[tuple]], iou_thr: float = 0.5) -> dict:
    """Single-class VOC-style average precision (all-points interpolation) at an IoU
    threshold. Detections are ranked globally by score; each GT box is matchable once.
    Used by A3 (mAP@0.5 has one class here, so mAP == AP)."""
    from bball.detect.interfaces import iou as _iou

    entries = []  # (score, frame, box)
    n_gt = 0
    for f, (preds, scores, gts) in enumerate(zip(pred_boxes, pred_scores, gt_boxes)):
        n_gt += len(gts)
        for box, sc in zip(preds, scores):
            entries.append((float(sc), f, box))
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "n_pred": len(entries)}
    entries.sort(key=lambda e: -e[0])
    matched = {f: set() for f in range(len(gt_boxes))}
    tp = np.zeros(len(entries))
    fp = np.zeros(len(entries))
    for k, (_, f, box) in enumerate(entries):
        best_j, best_i = -1, iou_thr
        for j, g in enumerate(gt_boxes[f]):
            if j in matched[f]:
                continue
            v = _iou(box, g)
            if v >= best_i:
                best_j, best_i = j, v
        if best_j >= 0:
            matched[f].add(best_j)
            tp[k] = 1
        else:
            fp[k] = 1
    if not len(entries):
        return {"ap": 0.0, "n_gt": n_gt, "n_pred": 0}
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # all-points interpolation: AP = integral of the monotone-max precision envelope.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "n_gt": n_gt, "n_pred": len(entries)}


def _frame_matches(gt_frame: dict, pred_frame: dict, iou_thr: float):
    """Hungarian IoU matching within one frame. Returns list of (gt_id, pred_id) at IoU>=thr."""
    from scipy.optimize import linear_sum_assignment

    from bball.detect.interfaces import iou as _iou

    gids, pids = list(gt_frame), list(pred_frame)
    if not gids or not pids:
        return []
    M = np.zeros((len(gids), len(pids)))
    for i, g in enumerate(gids):
        for j, p in enumerate(pids):
            M[i, j] = _iou(gt_frame[g], pred_frame[p])
    row, col = linear_sum_assignment(-M)
    return [(gids[r], pids[c]) for r, c in zip(row, col) if M[r, c] >= iou_thr]


def id_switches(gt_frames: list[dict], pred_frames: list[dict], iou_thr: float = 0.5) -> int:
    """CLEAR-MOT identity switches: a GT identity's matched track id changes across frames
    (counting only when it was previously matched to a *different* still-remembered id)."""
    last = {}  # gt_id -> pred_id last time it was matched
    sw = 0
    for gt_f, pr_f in zip(gt_frames, pred_frames):
        for g, p in _frame_matches(gt_f, pr_f, iou_thr):
            if g in last and last[g] != p:
                sw += 1
            last[g] = p
    return sw


def idf1(gt_frames: list[dict], pred_frames: list[dict], iou_thr: float = 0.5) -> dict:
    """Ristani IDF1: global bipartite identity matching maximising identity true positives.
    IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)."""
    from scipy.optimize import linear_sum_assignment

    from bball.detect.interfaces import iou as _iou

    gt_ids = sorted({g for f in gt_frames for g in f})
    pr_ids = sorted({p for f in pred_frames for p in f})
    gt_count = {g: sum(g in f for f in gt_frames) for g in gt_ids}
    pr_count = {p: sum(p in f for f in pred_frames) for p in pr_ids}
    n_gt_boxes = sum(gt_count.values())
    n_pr_boxes = sum(pr_count.values())
    if not gt_ids or not pr_ids:
        return {"idf1": 0.0, "idtp": 0, "idfp": n_pr_boxes, "idfn": n_gt_boxes}
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: j for j, p in enumerate(pr_ids)}
    cooc = np.zeros((len(gt_ids), len(pr_ids)))  # frames both present AND IoU>=thr
    for gt_f, pr_f in zip(gt_frames, pred_frames):
        for g, gb in gt_f.items():
            for p, pb in pr_f.items():
                if _iou(gb, pb) >= iou_thr:
                    cooc[gi[g], pi[p]] += 1
    row, col = linear_sum_assignment(-cooc)
    idtp = int(sum(cooc[r, c] for r, c in zip(row, col)))
    idfp = n_pr_boxes - idtp
    idfn = n_gt_boxes - idtp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": (2 * idtp / denom) if denom else 0.0, "idtp": idtp, "idfp": idfp, "idfn": idfn}


def hota_simplified(gt_frames: list[dict], pred_frames: list[dict], iou_thr: float = 0.5) -> dict:
    """SIMPLIFIED HOTA at a single localisation threshold (detection-matched level).

    HOTA = sqrt(DetA · AssA). DetA = |TP| / (|TP|+|FN|+|FP|); AssA averages the per-TP
    association accuracy A(c) = TPA / (TPA+FNA+FPA) over the standard HOTA co-occurrence
    counts. SIMPLIFICATION vs the reference metric: a single IoU threshold (0.5) instead of
    the 0.05..0.95 average, and greedy-Hungarian per-frame matching. Documented in the A4
    report; adequate for a synthetic tracker comparison, not a leaderboard submission."""
    matches = [_frame_matches(gt_f, pr_f, iou_thr) for gt_f, pr_f in zip(gt_frames, pred_frames)]
    tp = sum(len(m) for m in matches)
    n_gt = sum(len(f) for f in gt_frames)
    n_pr = sum(len(f) for f in pred_frames)
    fn = n_gt - tp
    fp = n_pr - tp
    det_a = tp / max(tp + fn + fp, 1e-9)
    # association counts over matched (gt_id, pred_id) pairs
    pair_c, g_c, p_c = {}, {}, {}
    for m in matches:
        for g, p in m:
            pair_c[(g, p)] = pair_c.get((g, p), 0) + 1
            g_c[g] = g_c.get(g, 0) + 1
            p_c[p] = p_c.get(p, 0) + 1
    if tp == 0:
        return {"hota": 0.0, "det_a": det_a, "ass_a": 0.0, "tp": tp, "fp": fp, "fn": fn}
    ass_sum = 0.0
    for m in matches:
        for g, p in m:
            tpa = pair_c[(g, p)]
            fna = g_c[g] - tpa
            fpa = p_c[p] - tpa
            ass_sum += tpa / max(tpa + fna + fpa, 1e-9)
    ass_a = ass_sum / tp
    return {"hota": float(np.sqrt(det_a * ass_a)), "det_a": float(det_a), "ass_a": float(ass_a),
            "tp": tp, "fp": fp, "fn": fn}


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
