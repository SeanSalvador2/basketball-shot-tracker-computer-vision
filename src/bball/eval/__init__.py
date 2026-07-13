"""EVAL — metrics, slicing, uncertainty, galleries, splits.

metrics.py    event P/R/F1 (+-0.25s), FP/hour, T3 cm error, zone confusion, ECE/Brier,
              track completeness, per-axis T5 accuracy
stratify.py   slice by scene-config axes
bootstrap.py  per-session bootstrap CIs
galleries.py  failure contact sheets with trajectory overlays
splits.py     session/scene-config split discipline + val-tune/val-cal (R6) + leakage test
"""
