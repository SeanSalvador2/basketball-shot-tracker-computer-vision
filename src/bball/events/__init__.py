"""CLASSIFY — event logic on top of geometry.

release.py         release detection (ball-wrist separation; fallback ball-above-head)
fsm.py             rim-normalized shot FSM, terminal-state MADE logic, margin score
miss_direction.py  rim-local left/right + short/long decomposition, per-axis confidence
calibration.py     temperature + Platt scaling, reliability-diagram utils

Predicates are rim-normalized (review R2): the annotated rim ellipse is the projective
image of the rim circle, so "inside the rim" is a fraction of its axes — placement-
transferable without unobservable 3D. The verdict is the terminal state, not the first
crossing (rattle-in and shooter's-roll resolve correctly).
"""
