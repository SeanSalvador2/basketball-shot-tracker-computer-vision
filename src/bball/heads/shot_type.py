"""T4 shot-type head: pull-up vs catch-and-shoot (plan §5.4, ablation A10).

Status: **harness-validated scaffold (regime S)** — trained/tested here only on synthetic
ball-height sequences to prove the plumbing; real accuracy claims arrive with Stage-B pose +
ball tracks. Three rungs, cheapest first (plan §4 baseline discipline):

1. `dribble_features` — the signal-processing feature the survey argued makes this a
   *feature, not a network*: FFT band energy at 1–3 Hz on ball height over the pre-release
   window (dribbling is a ~2 Hz vertical oscillation reaching near the floor) + floor-
   proximity minima count + possession duration.
2. `TwoFeatureLogistic` — logistic regression on [band energy, floor-minima rate]; if this
   hits ~80%+ (it does, trivially, on synthetic), any temporal net must justify itself.
3. `TemporalConv1D` — a small 1D-CNN over the raw height sequence, CPU-trainable, for the
   cases where hand-designed features saturate (Stage B decides with A10).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Dribble-oscillation features
# --------------------------------------------------------------------------- #
def dribble_features(ball_height_m: np.ndarray, fps: float, *, floor_thresh_m: float = 0.4,
                     band_hz: tuple[float, float] = (1.0, 3.0)) -> dict:
    """Features over a pre-release window of ball heights (metres, court frame — via the
    homography floor line or, degraded, normalized image height)."""
    z = np.asarray(ball_height_m, float)
    z = z[np.isfinite(z)]
    n = len(z)
    if n < 4:
        return {"band_energy": 0.0, "floor_minima_rate": 0.0, "duration_s": n / fps}
    zc = z - z.mean()
    spec = np.abs(np.fft.rfft(zc)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    total = float(spec[1:].sum()) + 1e-12          # exclude DC
    band_energy = float(spec[band].sum()) / total  # fraction of variance in the dribble band
    # floor-proximity minima: local minima below the floor threshold
    minima = 0
    for i in range(1, n - 1):
        if z[i] < z[i - 1] and z[i] <= z[i + 1] and z[i] < floor_thresh_m:
            minima += 1
    duration_s = n / fps
    return {"band_energy": band_energy, "floor_minima_rate": minima / max(duration_s, 1e-6),
            "duration_s": duration_s}


def feature_vector(ball_height_m: np.ndarray, fps: float) -> np.ndarray:
    f = dribble_features(ball_height_m, fps)
    return np.array([f["band_energy"], f["floor_minima_rate"]])


# --------------------------------------------------------------------------- #
# 2. Two-feature logistic baseline (own implementation — no sklearn dependency)
# --------------------------------------------------------------------------- #
@dataclass
class TwoFeatureLogistic:
    w: np.ndarray | None = None
    b: float = 0.0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, lr: float = 0.5, epochs: int = 500,
            l2: float = 1e-3) -> "TwoFeatureLogistic":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Xn = (X - self.mean_) / self.std_
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(epochs):
            p = self._sigmoid(Xn @ self.w + self.b)
            g_w = Xn.T @ (p - y) / len(y) + l2 * self.w
            g_b = float((p - y).mean())
            self.w -= lr * g_w
            self.b -= lr * g_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X, float) - self.mean_) / self.std_
        return self._sigmoid(Xn @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# --------------------------------------------------------------------------- #
# 3. Small 1D-CNN temporal head (torch; lazy import)
# --------------------------------------------------------------------------- #
def build_temporal_cnn(in_channels: int = 1, base_ch: int = 8, n_classes: int = 2):
    import torch.nn as nn

    class TemporalConv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, base_ch, 5, padding=2), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(base_ch, 2 * base_ch, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(2 * base_ch, n_classes)

        def forward(self, x):              # x: (B, C, T)
            h = self.net(x).squeeze(-1)
            return self.fc(h)

    return TemporalConv1D()


def train_temporal_cnn(model, X: np.ndarray, y: np.ndarray, *, epochs: int = 30,
                       lr: float = 1e-2, seed: int = 0) -> list[float]:
    """Reduced-scale CPU training on (N, C, T) sequences. Returns per-epoch losses."""
    import torch

    torch.manual_seed(seed)
    Xt = torch.from_numpy(np.asarray(X, np.float32))
    yt = torch.from_numpy(np.asarray(y, np.int64))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = torch.nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = lossfn(out, yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def predict_temporal_cnn(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.asarray(X, np.float32)))
        return out.argmax(dim=1).numpy()


# --------------------------------------------------------------------------- #
# Synthetic sequence generator (harness validation only)
# --------------------------------------------------------------------------- #
def synth_prerelease_sequences(n_per_class: int, *, fps: float = 60.0, dur_s: float = 2.0,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(X:(N, T) ball heights, y: 1=pull-up/dribble, 0=catch-and-shoot). Same oscillation
    model as synth.physics possession phase, plus hold jitter."""
    rng = np.random.default_rng(seed)
    T = int(dur_s * fps)
    t = np.arange(T) / fps
    X, y = [], []
    for _ in range(n_per_class):
        hold = rng.uniform(1.6, 2.0)
        f = rng.uniform(1.6, 2.4)               # dribble frequency (Hz)
        osc = 1.0 - np.abs(np.sin(np.pi * f * t + rng.uniform(0, np.pi)))
        z = 0.15 + osc * (hold - 0.15) + rng.normal(0, 0.02, T)
        X.append(z); y.append(1)                # pull-up
        z2 = hold + rng.normal(0, 0.02, T) + 0.05 * np.sin(2 * np.pi * 0.3 * t)  # sway, no dribble
        X.append(z2); y.append(0)               # catch-and-shoot
    return np.array(X), np.array(y)
