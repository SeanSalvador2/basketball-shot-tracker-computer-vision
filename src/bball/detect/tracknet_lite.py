"""TrackNet-lite — a small 3-frame heatmap UNet for the A1/A2 ablation arms.

The TrackNet lineage stacks N consecutive frames and regresses a Gaussian heatmap of the
ball. Phase-0 argued this is *not necessary* at basketball scale (20-40 px ball) the way it
is for a few-pixel shuttlecock; A1 turns that into an empirical question by pitting this
heatmap-temporal arm against per-frame bbox + ballistic bridging. This is a deliberately
tiny, CPU-trainable net (reduced scale — every number it produces is labelled S,
reduced-scale). Configs scale up unchanged on a GPU (Stage B).

Pure torch; imported lazily so geometry/synth code never pays the torch import cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian_heatmap(h: int, w: int, center, sigma: float = 3.0) -> np.ndarray:
    """A single-peak Gaussian heatmap (0..1). center in (x, y) heatmap pixels or None."""
    hm = np.zeros((h, w), np.float32)
    if center is None or (isinstance(center, float) and np.isnan(center)):
        return hm
    cx, cy = float(center[0]), float(center[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return hm
    ys, xs = np.mgrid[0:h, 0:w]
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    return hm


@dataclass
class TrackNetConfig:
    in_frames: int = 3           # A2 ablation: {1, 3, 5}
    input_h: int = 128
    input_w: int = 224
    base_ch: int = 16
    sigma: float = 3.0


def build_model(cfg: TrackNetConfig):
    import torch.nn as nn

    C = cfg.base_ch

    def block(cin, cout):
        # GroupNorm (not BatchNorm): batch-independent and identical in train/eval, which
        # matters for the small batches / reduced-scale training this net runs at.
        g = min(8, cout)
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
        )

    class TrackNetLite(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(cfg.in_frames, C)
            self.enc2 = block(C, 2 * C)
            self.pool = nn.MaxPool2d(2)
            self.bott = block(2 * C, 4 * C)
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec2 = block(4 * C + 2 * C, 2 * C)
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec1 = block(2 * C + C, C)
            self.head = nn.Conv2d(C, 1, 1)

        def forward(self, x):
            import torch

            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b = self.bott(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return torch.sigmoid(self.head(d1))

    return TrackNetLite()


def frames_to_input(frames_gray: list[np.ndarray], idx: int, cfg: TrackNetConfig) -> np.ndarray:
    """Stack `in_frames` grayscale frames ending at idx, resized to (input_h, input_w)."""
    import cv2

    stack = []
    for k in range(cfg.in_frames - 1, -1, -1):
        j = max(idx - k, 0)
        g = frames_gray[j]
        g = cv2.resize(g, (cfg.input_w, cfg.input_h)).astype(np.float32) / 255.0
        stack.append(g)
    return np.stack(stack, axis=0)


def build_training_tensors(clips: list[dict], cfg: TrackNetConfig):
    """clips: list of {'frames_gray': [HxW], 'centers_px': (N,2) in frame px, 'frame_hw': (H,W)}.
    Returns torch tensors (X: B x in_frames x h x w, Y: B x 1 x h x w)."""
    import torch

    Xs, Ys = [], []
    for clip in clips:
        frames = clip["frames_gray"]
        centers = clip["centers_px"]
        H, W = clip["frame_hw"]
        sx, sy = cfg.input_w / W, cfg.input_h / H
        for i in range(len(frames)):
            Xs.append(frames_to_input(frames, i, cfg))
            c = centers[i]
            hm_center = None if (c is None or np.isnan(np.asarray(c)).any()) else (c[0] * sx, c[1] * sy)
            Ys.append(gaussian_heatmap(cfg.input_h, cfg.input_w, hm_center, cfg.sigma)[None])
    X = torch.from_numpy(np.stack(Xs)).float()
    Y = torch.from_numpy(np.stack(Ys)).float()
    return X, Y


def train_tracknet(model, X, Y, *, epochs: int = 8, lr: float = 1e-3, batch: int = 16,
                   seed: int = 0, log_every: int = 2, logger=None):
    """Reduced-scale CPU training. Returns per-epoch loss list."""
    import torch

    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.MSELoss()
    n = X.shape[0]
    losses = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            pred = model(X[idx])
            loss = lossfn(pred, Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        ep_loss = tot / n
        losses.append(ep_loss)
        if logger and (ep % log_every == 0 or ep == epochs - 1):
            logger.info("tracknet epoch %d/%d loss=%.5f", ep + 1, epochs, ep_loss)
    return losses


def infer_ball(model, frames_gray: list[np.ndarray], idx: int, cfg: TrackNetConfig,
               frame_hw: tuple[int, int], *, peak_thresh: float = 0.3):
    """Return (xy_in_frame_px or None, score) for frame idx."""
    import torch

    x = torch.from_numpy(frames_to_input(frames_gray, idx, cfg))[None].float()
    model.eval()
    with torch.no_grad():
        hm = model(x)[0, 0].cpu().numpy()
    peak = float(hm.max())
    if peak < peak_thresh:
        return None, peak
    yy, xx = np.unravel_index(int(hm.argmax()), hm.shape)
    H, W = frame_hw
    xy = np.array([xx * W / cfg.input_w, yy * H / cfg.input_h])
    return xy, peak
