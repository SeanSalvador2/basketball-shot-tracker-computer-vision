"""Data association: IoU gating + Hungarian assignment + ByteTrack-style recovery.

Players get tracking-by-detection with a constant-velocity KF and IoU/Hungarian assignment
(plan §5.2). The one add-on that pays at <=3 targets is ByteTrack's idea: associate
*low-confidence* detections to surviving tracks before killing them (a half-occluded player
still yields a weak box). We skip ReID/GMC (no crowds, no camera motion — TrackID3x3
evidence). The ball is NEVER handed to this tracker (track-kill-on-miss is wrong under rim
occlusion); its own gated NN lives here as `gated_nearest` for possession mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from bball.detect.interfaces import Detection, iou
from bball.track.kalman import TrackState, cv_kalman_2d


def iou_matrix(boxes_a: list[tuple], boxes_b: list[tuple]) -> np.ndarray:
    M = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            M[i, j] = iou(a, b)
    return M


def associate_iou(track_boxes: list[tuple], det_boxes: list[tuple], iou_threshold: float = 0.2):
    """Hungarian assignment on IoU. Returns (matches[(ti,di)], unmatched_tracks, unmatched_dets)."""
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    M = iou_matrix(track_boxes, det_boxes)
    row, col = linear_sum_assignment(-M)             # maximize IoU
    matches, ut, ud = [], [], []
    matched_t, matched_d = set(), set()
    for r, c in zip(row, col):
        if M[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
    ut = [i for i in range(len(track_boxes)) if i not in matched_t]
    ud = [j for j in range(len(det_boxes)) if j not in matched_d]
    return matches, ut, ud


def gated_nearest(pred_xy: np.ndarray, candidates, gate_px: float):
    """Ball possession-mode association: nearest candidate within a gate. Returns the
    BallCandidate (or None). candidates is an iterable of BallCandidate."""
    best, best_d = None, gate_px
    for c in candidates:
        if c.xy is None:
            continue
        d = float(np.hypot(*(c.xy - pred_xy)))
        if d <= best_d:
            best, best_d = c, d
    return best


@dataclass
class ByteTrackConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    iou_match: float = 0.2
    max_age: int = 30            # frames a track survives without a match
    min_hits: int = 2
    process_std: float = 60.0
    meas_std: float = 4.0
    dt: float = 1.0


class ByteTrackPlayerTracker:
    """Multi-object player tracker: CV Kalman + IoU/Hungarian + low-score recovery."""

    def __init__(self, config: ByteTrackConfig | None = None):
        self.cfg = config or ByteTrackConfig()
        self.tracks: list[TrackState] = []
        self._next_id = 0
        self._sizes: dict[int, tuple[float, float]] = {}

    def _predicted_box(self, tr: TrackState) -> tuple:
        cx, cy = tr.position
        w, h = self._sizes.get(tr.track_id, (30.0, 60.0))
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _new_track(self, det: Detection) -> None:
        kf = cv_kalman_2d(det.center, dt=self.cfg.dt, process_std=self.cfg.process_std,
                          meas_std=self.cfg.meas_std)
        tr = TrackState(track_id=self._next_id, kf=kf)
        self._sizes[self._next_id] = det.wh
        self.tracks.append(tr)
        self._next_id += 1

    def update(self, detections: list[Detection]) -> dict[int, tuple]:
        cfg = self.cfg
        for tr in self.tracks:
            tr.kf.predict()
            tr.age += 1
            tr.time_since_update += 1

        high = [d for d in detections if d.score >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.score < cfg.high_thresh]

        track_boxes = [self._predicted_box(tr) for tr in self.tracks]
        # -- pass 1: high-score detections --
        matches, ut, ud = associate_iou(track_boxes, [d.bbox for d in high], cfg.iou_match)
        for ti, di in matches:
            self._apply_match(self.tracks[ti], high[di])

        # -- pass 2: low-score detections to still-unmatched tracks (ByteTrack recovery) --
        rem_track_boxes = [track_boxes[i] for i in ut]
        if rem_track_boxes and low:
            m2, ut2, _ = associate_iou(rem_track_boxes, [d.bbox for d in low], cfg.iou_match)
            matched_rem = set()
            for ri, di in m2:
                self._apply_match(self.tracks[ut[ri]], low[di])
                matched_rem.add(ri)
            still_unmatched = [ut[i] for i in range(len(ut)) if i not in matched_rem]
        else:
            still_unmatched = ut

        # -- unmatched high detections -> new tracks --
        for di in ud:
            self._new_track(high[di])

        # -- cull stale tracks --
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= cfg.max_age]

        out = {}
        for tr in self.tracks:
            if tr.hits >= cfg.min_hits or tr.time_since_update == 0:
                out[tr.track_id] = self._predicted_box(tr)
        return out

    def _apply_match(self, tr: TrackState, det: Detection) -> None:
        tr.kf.update(det.center)
        tr.hits += 1
        tr.time_since_update = 0
        # EMA the box size.
        ow, oh = self._sizes.get(tr.track_id, det.wh)
        nw, nh = det.wh
        self._sizes[tr.track_id] = (0.7 * ow + 0.3 * nw, 0.7 * oh + 0.3 * nh)
        tr.history.append((det.frame_idx, tr.position.copy()))


def run_player_tracker(frame_detections: list[list[Detection]],
                       config: ByteTrackConfig | None = None) -> list[dict[int, tuple]]:
    """Run the tracker over a whole clip's per-frame detections. Returns per-frame
    {track_id: box}."""
    tracker = ByteTrackPlayerTracker(config)
    return [tracker.update(dets) for dets in frame_detections]
