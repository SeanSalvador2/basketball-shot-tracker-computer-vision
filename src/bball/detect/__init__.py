"""DETECT — per-frame ball / player / rim candidates.

interfaces.py           Detection, BallCandidate dataclasses
torchvision_detector.py fasterrcnn (mobilenet_v3 / resnet50_v2), person+sports-ball
bgsub.py                MOG2 + size/shape/temporal filtering (classical baseline, R4)
tracknet_lite.py        small 3-frame heatmap UNet (trainable at 512px on CPU, A1/A2)

Rim is not a COCO class -> per-session manual ROI/ellipse (documented as deliberate).
"""
