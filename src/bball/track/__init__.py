"""TRACK — temporal association + ballistic occlusion bridging.

kalman.py       constant-velocity KF (own implementation, tested vs analytic cases)
association.py  Hungarian (scipy) + IoU gating + ByteTrack-style low-score recovery
ballistic.py    two-level trajectory model (L1 image-space quadratic, L2 constrained 3D)

The ball is never handed to a SORT-family tracker (track-kill-on-miss is exactly wrong
under rim occlusion). A homography cannot lift an airborne ball, and a 3D parabola does
not project to an exact image parabola — hence the two levels, each used only for what it
can support (review R1).
"""
from bball.track.association import (
    ByteTrackConfig,
    ByteTrackPlayerTracker,
    associate_iou,
    gated_nearest,
    iou_matrix,
    run_player_tracker,
)
from bball.track.ballistic import (
    BridgeResult,
    Level1Fit,
    Trajectory3D,
    bridge_trajectory,
    fit_level1,
    miss_vector_rim_local,
    reconstruct_flight_3d,
)
from bball.track.kalman import KalmanFilter, TrackState, cv_kalman_2d

__all__ = [
    "KalmanFilter", "cv_kalman_2d", "TrackState",
    "iou_matrix", "associate_iou", "gated_nearest",
    "ByteTrackConfig", "ByteTrackPlayerTracker", "run_player_tracker",
    "Level1Fit", "fit_level1", "bridge_trajectory", "BridgeResult",
    "Trajectory3D", "reconstruct_flight_3d", "miss_vector_rim_local",
]
