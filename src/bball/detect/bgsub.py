"""Background-subtraction ball detector — the classical baseline (review R4).

The fixed camera is the strongest classical prior available: the moving ball is the
dominant small mover at wide framing, so MOG2 background subtraction + size/shape/temporal
filtering is a legitimate zero-weights ball-candidate generator. It is (a) a detection
baseline and (b) a candidate-proposal channel that fuses with a neural detector (A1's
third arm). Registered hypothesis (A1): high recall on clean flight, fails under occlusion
and multi-mover chaos; fusion inherits the recall without the failure.

Runs directly on rendered frames — no torch, no weights. Candidate coordinates are in the
input frame's pixel space; the caller rescales to native camera pixels if the frame was
downscaled for rendering.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bball.detect.interfaces import BallCandidate


@dataclass
class BgSubConfig:
    history: int = 120
    var_threshold: float = 25.0
    # Shadow detection off by default: MOG2 flags darker-than-background movers as "shadow"
    # (value 127), and a ball is often darker than a bright floor in grayscale, so leaving
    # it on silently discards the ball. Synthetic renders have no cast shadows anyway.
    detect_shadows: bool = False
    min_area: float = 12.0
    max_area: float = 2500.0          # players/large blobs excluded
    min_circularity: float = 0.45     # motion blur lowers this, so not too strict
    max_aspect: float = 3.5
    morph_kernel: int = 3
    learning_rate: float = -1.0       # MOG2 auto


class BgSubBallDetector:
    def __init__(self, config: BgSubConfig | None = None):
        self.cfg = config or BgSubConfig()
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history, varThreshold=self.cfg.var_threshold,
            detectShadows=self.cfg.detect_shadows)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_kernel,) * 2)

    def _candidates_from_mask(self, mask: np.ndarray, frame_idx: int) -> list[BallCandidate]:
        # Shadows are marked 127 by MOG2; keep only hard foreground.
        fg = (mask >= 200).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[BallCandidate] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.cfg.min_area <= area <= self.cfg.max_area):
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 1e-3:
                continue
            circ = 4 * np.pi * area / (perim * perim)
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if circ < self.cfg.min_circularity or aspect > self.cfg.max_aspect:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            radius = float(np.sqrt(area / np.pi))
            # score = how ball-like (circularity), the arbitration signal for fusion.
            out.append(BallCandidate(frame_idx=frame_idx, xy=np.array([cx, cy]),
                                     score=float(np.clip(circ, 0, 1)), radius_px=radius,
                                     source="bgsub", bbox=(float(x), float(y), float(x + w), float(y + h))))
        return out

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int = -1) -> list[BallCandidate]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = self._sub.apply(gray, learningRate=self.cfg.learning_rate)
        return self._candidates_from_mask(mask, frame_idx)

    def process_stream(self, frames: list[np.ndarray], *, warmup: int = 5,
                       temporal: bool = True) -> list[list[BallCandidate]]:
        """Per-frame candidates for a clip. `warmup` frames prime the background model.
        `temporal` keeps only candidates with a plausible neighbour in an adjacent frame
        (suppresses one-frame blips) while never requiring persistence for fast motion."""
        per_frame: list[list[BallCandidate]] = []
        for i, f in enumerate(frames):
            cands = self.process_frame(f, i)
            per_frame.append(cands)
        if not temporal:
            return per_frame
        return self._temporal_filter(per_frame)

    @staticmethod
    def _temporal_filter(per_frame: list[list[BallCandidate]], max_disp: float = 120.0) -> list[list[BallCandidate]]:
        """Drop isolated single-frame candidates that have no plausible neighbour within
        max_disp in the previous or next frame."""
        n = len(per_frame)
        kept: list[list[BallCandidate]] = [[] for _ in range(n)]
        for i in range(n):
            for c in per_frame[i]:
                if c.xy is None:
                    continue
                has_neighbor = False
                for j in (i - 1, i + 1):
                    if 0 <= j < n:
                        for c2 in per_frame[j]:
                            if c2.xy is not None and np.hypot(*(c.xy - c2.xy)) <= max_disp:
                                has_neighbor = True
                                break
                    if has_neighbor:
                        break
                if has_neighbor or n <= 2:
                    kept[i].append(c)
        return kept


def fuse_candidates(neural: list[BallCandidate], classical: list[BallCandidate],
                    *, merge_dist: float = 20.0) -> list[BallCandidate]:
    """Union of candidate channels; where both fire near the same point, keep the higher
    score (the detector arbitrates). This is A1's fusion arm."""
    fused = list(neural)
    for c in classical:
        if c.xy is None:
            continue
        dup = False
        for f in fused:
            if f.xy is not None and np.hypot(*(c.xy - f.xy)) <= merge_dist:
                dup = True
                if c.score > f.score:
                    f.score = c.score
                break
        if not dup:
            fused.append(c)
    return fused
