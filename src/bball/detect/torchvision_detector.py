"""Stage-A bootstrap detector: torchvision COCO-pretrained Faster R-CNN (BSD-3).

Two backbones per plan §5.1: `fasterrcnn_mobilenet_v3_large_fpn` (throughput) and
`fasterrcnn_resnet50_fpn_v2` (accuracy reference). COCO gives `person` + `sports ball`
for free; rim is not a COCO class (handled by per-session ROI, lift.rim_frame).

WEIGHTS NOTE (documented deviation): torchvision's COCO weights are hosted on
download.pytorch.org, which the Stage-A build container firewalls (403). The wrapper is
therefore weight-agnostic: it *tries* to load pretrained weights and, if the download is
blocked, constructs the architecture with random weights and flags `pretrained=False` so
the plumbing (resolution control, class mapping, Detection emission) is still exercised
and smoke-tested. On a normal machine / Stage B the same call loads real COCO weights and
becomes a genuine detection baseline. Input resolution is configurable for the A3 ablation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bball.detect.interfaces import Detection
from bball.utils.logging import get_logger

log = get_logger("bball.detect.torchvision")

# COCO (91-class) category ids used by torchvision detection heads.
COCO_PERSON = 1
COCO_SPORTS_BALL = 37


@dataclass
class DetectorConfig:
    backbone: str = "mobilenet"       # mobilenet | resnet50
    score_thresh: float = 0.3
    min_size: int = 768               # A3 resolution lever: {512, 768, 1088}
    max_size: int = 1333
    ball_score_thresh: float = 0.2    # loose box that keeps track continuity > tight flicker
    device: str = "cpu"


class TorchvisionBallPlayerDetector:
    def __init__(self, config: DetectorConfig | None = None, *, try_pretrained: bool = True):
        self.cfg = config or DetectorConfig()
        self.pretrained = False
        self._model = None
        self._build(try_pretrained)

    def _build(self, try_pretrained: bool) -> None:
        import torch
        import torchvision
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_fpn,
            fasterrcnn_resnet50_fpn_v2,
        )

        torch.set_num_threads(max(1, torch.get_num_threads()))
        ctor = (fasterrcnn_mobilenet_v3_large_fpn if self.cfg.backbone == "mobilenet"
                else fasterrcnn_resnet50_fpn_v2)
        kwargs = dict(min_size=self.cfg.min_size, max_size=self.cfg.max_size)
        if try_pretrained:
            try:
                self._model = ctor(weights="DEFAULT", **kwargs)
                self.pretrained = True
                log.info("loaded COCO-pretrained %s", self.cfg.backbone)
            except Exception as e:  # download blocked / offline
                log.warning("COCO weights unavailable (%s); using random init (plumbing only). "
                            "Stage B / a networked machine loads real weights.", type(e).__name__)
                self._model = ctor(weights=None, weights_backbone=None, num_classes=91, **kwargs)
        else:
            self._model = ctor(weights=None, weights_backbone=None, num_classes=91, **kwargs)
        self._model.eval().to(self.cfg.device)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int = -1) -> list[Detection]:
        """Run detection on one BGR frame; return ball + person Detections above threshold."""
        import torch

        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        with torch.no_grad():
            out = self._model([tensor.to(self.cfg.device)])[0]
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        dets: list[Detection] = []
        for box, lab, sc in zip(boxes, labels, scores):
            if lab == COCO_SPORTS_BALL and sc >= self.cfg.ball_score_thresh:
                dets.append(Detection(bbox=tuple(map(float, box)), score=float(sc), label="ball", frame_idx=frame_idx))
            elif lab == COCO_PERSON and sc >= self.cfg.score_thresh:
                dets.append(Detection(bbox=tuple(map(float, box)), score=float(sc), label="person", frame_idx=frame_idx))
        return dets

    def ball_candidates(self, frame_bgr: np.ndarray, frame_idx: int = -1):
        """Convenience: only the ball detections, as BallCandidates."""
        from bball.detect.interfaces import BallCandidate

        out = []
        for d in self.detect(frame_bgr, frame_idx):
            if d.label == "ball":
                w, h = d.wh
                out.append(BallCandidate(frame_idx=frame_idx, xy=d.center, score=d.score,
                                         radius_px=float((w + h) / 4.0), source="detector", bbox=d.bbox))
        return out
