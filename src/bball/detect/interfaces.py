"""Detection contracts shared across DETECT / TRACK / SYNTH.

Defined early (nominally an M4 file) because the synthetic detection-noise model emits
these same types — one canonical `BallCandidate`/`Detection` flows through the whole
pipeline whether it came from a real detector, background subtraction, a heatmap net, or
the synthetic noise model. That uniformity is what lets logic-level ablations run on
synthetic candidates and swap in a real detector unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Label = Literal["ball", "person", "rim"]
Source = Literal["detector", "bgsub", "heatmap", "gt", "synthetic"]


@dataclass
class Detection:
    """An axis-aligned box detection in image pixels."""

    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    score: float
    label: Label
    frame_idx: int = -1

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    @property
    def wh(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0, y1 - y0)

    @property
    def area(self) -> float:
        w, h = self.wh
        return max(w, 0.0) * max(h, 0.0)


@dataclass
class BallCandidate:
    """A ball position hypothesis for a single frame (or a missed/occluded frame)."""

    frame_idx: int
    xy: Optional[np.ndarray]           # image centre, or None when not observed this frame
    score: float = 0.0
    radius_px: float = 0.0
    source: Source = "synthetic"
    bbox: Optional[tuple[float, float, float, float]] = None
    occlusion: float = 0.0             # 0 = fully visible, 1 = fully occluded
    meta: dict = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.xy is not None


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    return inter / (area_a + area_b - inter + 1e-9)
