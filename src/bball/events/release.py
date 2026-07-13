"""Release detection, last-ground-contact feet read, and flight segmentation.

Release (plan §1.1): the first frame where ball-center-to-wrist distance exceeds 1.5x ball
radius with positive vertical ball velocity, falling back to ball-above-head + upward
velocity when pose is unavailable. All downstream tasks key off this timestamp, so its
tolerance is what the release accuracy metric measures.

For T3 (shot location) the read is the shooter's feet at the **last ground-contact frame
before release**, not at release — more accurate for jump shots and matching the basketball
semantics of where a shot was "from".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smooth_vy(ball_xy: np.ndarray, win: int = 3) -> np.ndarray:
    n = ball_xy.shape[0]
    vy = np.zeros(n)
    for i in range(1, n):
        j0 = max(0, i - win)
        seg = ball_xy[j0:i + 1, 1]
        if len(seg) >= 2 and not np.isnan(seg).any():
            vy[i] = (seg[-1] - seg[0]) / (len(seg) - 1)
        else:
            vy[i] = vy[i - 1]
    return vy


def detect_release_pose(ball_xy: np.ndarray, wrist_xy: np.ndarray, ball_radius_px,
                        *, sep_mult: float = 1.5) -> int:
    """Pose-based: first frame where ball-wrist separation exceeds sep_mult x ball radius
    with the ball moving upward (image vy < 0). Returns frame index or -1."""
    xy = np.asarray(ball_xy, float)
    wr = np.asarray(wrist_xy, float)
    rad = np.broadcast_to(np.asarray(ball_radius_px, float), (xy.shape[0],))
    vy = _smooth_vy(xy)
    for i in range(1, xy.shape[0]):
        if np.isnan(xy[i]).any() or np.isnan(wr[i]).any():
            continue
        sep = np.hypot(*(xy[i] - wr[i]))
        if sep > sep_mult * rad[i] and vy[i] < 0:
            return i
    return -1


def detect_release_fallback(ball_xy: np.ndarray, *, head_y: float | None = None,
                            rise_vel_px: float = 1.5, win: int = 3) -> int:
    """Pose-free: first frame with sustained upward velocity (and, if a head line is given,
    the ball above the head). Returns frame index or -1."""
    xy = np.asarray(ball_xy, float)
    vy = _smooth_vy(xy, win)
    for i in range(1, xy.shape[0]):
        if np.isnan(xy[i]).any():
            continue
        if vy[i] < -rise_vel_px and (head_y is None or xy[i, 1] < head_y):
            return i
    return -1


def last_ground_contact_frame(feet_height_m: np.ndarray, release_idx: int,
                              *, ground_thresh_m: float = 0.08) -> int:
    """Last frame at/near ground level before release (feet height ~0). For a jump shot the
    shooter has left the floor by release, so we read the location one push-off earlier."""
    fh = np.asarray(feet_height_m, float)
    for i in range(min(release_idx, len(fh) - 1), -1, -1):
        if fh[i] <= ground_thresh_m:
            return i
    return max(release_idx, 0)


@dataclass
class FlightSegmenter:
    rise_vel_px: float = 1.5
    min_flight_frames: int = 6
    win: int = 3

    def segment(self, ball_xy: np.ndarray, times: np.ndarray | None = None) -> list[tuple[int, int]]:
        """Split a continuous ball stream into flight (start, end) spans, one per release.
        A release is an upward-velocity onset; the flight runs to the next release (or end).
        Re-arms only once the ball has clearly settled (descending) to avoid re-triggering
        on rim bounces."""
        xy = np.asarray(ball_xy, float)
        n = xy.shape[0]
        vy = _smooth_vy(xy, self.win)
        releases: list[int] = []
        armed = True
        settle = 0
        for i in range(1, n):
            if armed and vy[i] < -self.rise_vel_px and vy[i - 1] >= -self.rise_vel_px:
                releases.append(i)
                armed = False
                settle = 0
            if vy[i] > 0.5:
                settle += 1
                if settle >= self.win:      # ball has been descending -> ready for a new shot
                    armed = True
            else:
                settle = 0
        segs = []
        for k, r in enumerate(releases):
            end = releases[k + 1] if k + 1 < len(releases) else n
            start = max(r - self.win, 0)
            if end - start >= self.min_flight_frames:
                segs.append((start, end))
        return segs
