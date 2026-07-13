"""DETECT — per-frame ball / player / rim candidates.

interfaces.py           Detection, BallCandidate dataclasses (shared contract)
torchvision_detector.py fasterrcnn (mobilenet_v3 / resnet50_v2), person+sports-ball
bgsub.py                MOG2 + size/shape/temporal filtering (classical baseline, R4)
tracknet_lite.py        small 3-frame heatmap UNet (trainable at reduced scale, A1/A2)

Rim is not a COCO class -> per-session manual ROI/ellipse (lift.rim_frame.RimAnnotation),
documented as a deliberate non-problem for the fixed camera.
"""
from bball.detect.bgsub import BgSubBallDetector, BgSubConfig, fuse_candidates
from bball.detect.interfaces import BallCandidate, Detection, iou

__all__ = [
    "Detection",
    "BallCandidate",
    "iou",
    "BgSubBallDetector",
    "BgSubConfig",
    "fuse_candidates",
]
