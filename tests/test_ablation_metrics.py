"""Tests for the detection (A3) and multi-object-tracking (A4) evaluation metrics."""
from __future__ import annotations

from bball.eval.metrics import (
    average_precision_at_iou,
    detection_recall_at_iou,
    hota_simplified,
    id_switches,
    idf1,
)


def _box(cx, cy, w=20, h=20):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# --------------------------------------------------------------------------- #
# Detection metrics (A3)
# --------------------------------------------------------------------------- #
def test_recall_perfect_and_empty():
    gt = [[_box(100, 100)], [_box(120, 100)]]
    assert detection_recall_at_iou(gt, gt, iou_thr=0.3)["recall"] == 1.0
    empty = [[], []]
    assert detection_recall_at_iou(empty, gt, iou_thr=0.3)["recall"] == 0.0


def test_recall_iou_threshold():
    gt = [[_box(100, 100, 20, 20)]]
    # a box shifted by 10px overlaps ~ IoU 0.33 -> recalled at 0.3, missed at 0.5
    pred = [[_box(110, 100, 20, 20)]]
    assert detection_recall_at_iou(pred, gt, iou_thr=0.3)["recall"] == 1.0
    assert detection_recall_at_iou(pred, gt, iou_thr=0.5)["recall"] == 0.0


def test_average_precision_perfect_vs_noise():
    gt = [[_box(100, 100)], [_box(120, 100)]]
    scores = [[0.9], [0.8]]
    ap = average_precision_at_iou(gt, scores, gt, iou_thr=0.5)["ap"]
    assert ap > 0.99
    # a confident false positive far from GT drags AP down
    pred = [[_box(100, 100), _box(500, 500)], [_box(120, 100)]]
    sc = [[0.9, 0.95], [0.8]]
    ap2 = average_precision_at_iou(pred, sc, gt, iou_thr=0.5)["ap"]
    assert ap2 < ap


# --------------------------------------------------------------------------- #
# Tracking metrics (A4)
# --------------------------------------------------------------------------- #
def _clean_scene(n=10):
    gt, pred = [], []
    for t in range(n):
        gt.append({0: _box(100 + t, 100), 1: _box(300 - t, 200)})
        pred.append({7: _box(100 + t, 100), 9: _box(300 - t, 200)})  # ids differ but consistent
    return gt, pred


def test_perfect_tracking_scores_one_zero_switches():
    gt, pred = _clean_scene()
    assert id_switches(gt, pred) == 0
    assert idf1(gt, pred)["idf1"] > 0.99
    h = hota_simplified(gt, pred)
    assert h["hota"] > 0.99 and h["det_a"] > 0.99 and h["ass_a"] > 0.99


def test_identity_swap_counts_switch_and_hurts_idf1():
    gt, pred = _clean_scene(10)
    # swap the two predicted ids for the second half -> one switch per GT id
    for t in range(5, 10):
        pred[t] = {9: gt[t][0], 7: gt[t][1]}
    assert id_switches(gt, pred) >= 2
    assert idf1(gt, pred)["idf1"] < 0.99
    assert hota_simplified(gt, pred)["ass_a"] < 0.99  # association degraded


def test_empty_predictions_zero():
    gt, _ = _clean_scene()
    empty = [{} for _ in gt]
    assert id_switches(gt, empty) == 0
    assert idf1(gt, empty)["idf1"] == 0.0
    assert hota_simplified(gt, empty)["hota"] == 0.0
