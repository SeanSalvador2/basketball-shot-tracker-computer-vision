"""CLASSIFY — event logic on top of geometry.

release.py         release detection (ball-wrist separation; fallback ball-above-head),
                   last-ground-contact feet read, flight segmentation
fsm.py             rim-normalized shot FSM, terminal-state MADE logic, margin score
miss_direction.py  rim-local left/right + short/long decomposition, per-axis confidence
calibration.py     temperature + Platt scaling, reliability-diagram / ECE / Brier utils
"""
from bball.events.calibration import (
    PlattScaler,
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from bball.events.fsm import FSMConfig, ShotEvent, ShotFSM, ShotOutcome, State, run_fsm_stream
from bball.events.miss_direction import MissDirectionResult, decompose_miss
from bball.events.release import (
    FlightSegmenter,
    detect_release_fallback,
    detect_release_pose,
    last_ground_contact_frame,
)

__all__ = [
    "ShotFSM", "FSMConfig", "ShotOutcome", "ShotEvent", "State", "run_fsm_stream",
    "detect_release_pose", "detect_release_fallback", "last_ground_contact_frame", "FlightSegmenter",
    "decompose_miss", "MissDirectionResult",
    "TemperatureScaler", "PlattScaler", "reliability_curve", "expected_calibration_error", "brier_score",
]
