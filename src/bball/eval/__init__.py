"""EVAL — metrics, slicing, uncertainty, galleries, splits.

metrics.py    event P/R/F1 (+-0.25s), FP/hour, T3 cm error, zone confusion,
              track completeness, per-axis T5 accuracy
stratify.py   slice by scene-config axes
bootstrap.py  per-session bootstrap CIs
galleries.py  failure contact sheets with trajectory overlays
splits.py     session/scene-config split discipline + val-tune/val-cal (R6) + leakage test
"""
from bball.eval.bootstrap import bootstrap_ci, paired_session_delta
from bball.eval.metrics import (
    PRF,
    event_prf,
    false_attempts_per_hour,
    outcome_prf,
    per_axis_t5_accuracy,
    t3_error_cm,
    track_completeness,
    zone_confusion,
)
from bball.eval.splits import Split, assert_no_leakage, assert_test_venue_held_out, make_split
from bball.eval.stratify import group_by, occlusion_bucket, stratified_metric

__all__ = [
    "PRF", "event_prf", "outcome_prf", "false_attempts_per_hour", "t3_error_cm",
    "zone_confusion", "track_completeness", "per_axis_t5_accuracy",
    "Split", "make_split", "assert_no_leakage", "assert_test_venue_held_out",
    "group_by", "stratified_metric", "occlusion_bucket",
    "bootstrap_ci", "paired_session_delta",
]
