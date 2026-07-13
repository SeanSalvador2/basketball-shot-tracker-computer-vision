"""T5 miss direction — rim-local decomposition with per-axis confidence (plan §5.4).

The miss vector at rim arrival splits into **left/right** (image-plane dominant, robust) and
**short/long** (depth-dominant, azimuth-dependent). Level-2 reconstruction (track.ballistic)
supplies both, but the depth axis is only trustworthy when the camera sees it — so short/long
is confidence-gated: the app reports it only when depth confidence clears a threshold, and
otherwise hides it rather than guessing. A6 turns the confidence-vs-azimuth relationship into
the product's camera-placement guidance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bball.track.ballistic import Trajectory3D, miss_vector_rim_local


@dataclass
class AxisCall:
    label: str
    confidence: float
    shown: bool


@dataclass
class MissDirectionResult:
    left_right: AxisCall
    short_long: AxisCall
    depth_m: float
    lateral_m: float
    multi_label: str        # e.g. "short-left"; omits a hidden axis

    def as_dict(self) -> dict:
        return {
            "left_right": {"label": self.left_right.label, "confidence": self.left_right.confidence,
                           "shown": self.left_right.shown},
            "short_long": {"label": self.short_long.label, "confidence": self.short_long.confidence,
                           "shown": self.short_long.shown},
            "depth_m": self.depth_m, "lateral_m": self.lateral_m, "multi_label": self.multi_label,
        }


def decompose_miss(
    traj3d: Trajectory3D,
    rim_center_3d,
    shooter_feet_xy,
    *,
    min_lateral_conf: float = 0.3,
    min_depth_conf: float = 0.4,
    dead_zone_m: float = 0.05,
) -> MissDirectionResult:
    """Decompose a reconstructed miss into per-axis calls. Axes inside the dead zone (a
    near-centred axis) or below their confidence threshold are not shown."""
    mv = miss_vector_rim_local(traj3d, rim_center_3d, shooter_feet_xy)
    lat, depth = mv["lateral_m"], mv["depth_m"]
    lr_conf, sl_conf = mv["lateral_confidence"], mv["depth_confidence"]

    lr_label = "" if abs(lat) < dead_zone_m else ("right" if lat > 0 else "left")
    sl_label = "" if abs(depth) < dead_zone_m else ("long" if depth > 0 else "short")
    lr_shown = bool(lr_label) and lr_conf >= min_lateral_conf
    sl_shown = bool(sl_label) and sl_conf >= min_depth_conf

    parts = []
    if sl_shown:
        parts.append(sl_label)
    if lr_shown:
        parts.append(lr_label)
    multi = "-".join(parts) if parts else "uncertain"

    return MissDirectionResult(
        left_right=AxisCall(lr_label, float(lr_conf), lr_shown),
        short_long=AxisCall(sl_label, float(sl_conf), sl_shown),
        depth_m=float(depth), lateral_m=float(lat), multi_label=multi)
