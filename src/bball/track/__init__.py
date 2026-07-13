"""TRACK — temporal association + ballistic occlusion bridging.

kalman.py       constant-velocity KF (own implementation, tested vs analytic cases)
association.py  Hungarian (scipy) + IoU gating + ByteTrack-style low-score second pass
ballistic.py    two-level trajectory model (L1 image-space quadratic, L2 constrained 3D)

Design note (plan §5.2 / review R1): the ball is never handed to a SORT-family tracker
(track-kill-on-miss is exactly wrong under rim occlusion). A homography cannot lift an
airborne ball, and a 3D parabola does not project to an exact image parabola — hence the
two levels, each used only for what it can support.
"""
