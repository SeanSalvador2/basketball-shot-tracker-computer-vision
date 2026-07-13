"""T6 audio scaffold: log-mel windowing at the vision-determined rim-arrival + an
embedding-head interface (plan §5.4).

Status: **scaffold only — no accuracy claims of any kind** (the plan's honesty boundary:
no academic prior art exists for basketball shot-outcome audio; T6 is a research angle,
not a commitment). What Stage A ships:

* `extract_rim_window` — cut the audio window around the *vision-determined* rim-arrival
  time (audio is a confirmatory channel gated by vision timing, never a free-running one).
* `log_mel` — dependency-free numpy STFT + mel filterbank (no librosa/torchaudio import
  for a scaffold).
* `AudioHead` — frozen-embedding + logistic-head interface; the embedding is injectable
  (Stage B: YAMNet/BEATs per the survey; here: `mel_stats_embedding` for plumbing tests).

Design notes carried from review R11: slow-mo retimes video but records audio at normal
rate, so audio-critical sessions capture at 60 fps normal speed, and every session opens
with a ball bounce — a natural clapperboard the impulse test below mirrors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Windowing (vision-gated)
# --------------------------------------------------------------------------- #
def extract_rim_window(waveform: np.ndarray, sr: int, rim_arrival_s: float,
                       *, pre_s: float = 0.15, post_s: float = 0.35) -> np.ndarray:
    """Cut [rim_arrival - pre, rim_arrival + post] with zero-padding at clip edges."""
    w = np.asarray(waveform, float)
    i0 = int(round((rim_arrival_s - pre_s) * sr))
    i1 = int(round((rim_arrival_s + post_s) * sr))
    out = np.zeros(i1 - i0)
    lo, hi = max(i0, 0), min(i1, len(w))
    if hi > lo:
        out[lo - i0: hi - i0] = w[lo:hi]
    return out


# --------------------------------------------------------------------------- #
# Log-mel (numpy-only)
# --------------------------------------------------------------------------- #
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, float) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, min(c, fb.shape[1])):
            if c > l:
                fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, min(r, fb.shape[1])):
            if r > c:
                fb[m - 1, k] = (r - k) / (r - c)
    return fb


def log_mel(waveform: np.ndarray, sr: int, *, n_fft: int = 512, hop: int = 128,
            n_mels: int = 40) -> np.ndarray:
    """(n_mels, n_frames) log-mel spectrogram via numpy STFT (Hann window)."""
    w = np.asarray(waveform, float)
    if len(w) < n_fft:
        w = np.pad(w, (0, n_fft - len(w)))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(w) - n_fft) // hop
    frames = np.stack([w[i * hop: i * hop + n_fft] * win for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (n_frames, n_fft//2+1)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = spec @ fb.T                                        # (n_frames, n_mels)
    return np.log(mel.T + 1e-10)                             # (n_mels, n_frames)


def mel_stats_embedding(logmel: np.ndarray) -> np.ndarray:
    """Plumbing embedding: per-band mean+std + temporal peak position. Stage B replaces
    this with a frozen YAMNet/BEATs embedding behind the same interface."""
    mu = logmel.mean(axis=1)
    sd = logmel.std(axis=1)
    peak_t = np.argmax(logmel.max(axis=0)) / max(logmel.shape[1] - 1, 1)
    return np.concatenate([mu, sd, [peak_t]])


# --------------------------------------------------------------------------- #
# Embedding + head interface
# --------------------------------------------------------------------------- #
@dataclass
class AudioHead:
    """Frozen embedding fn + tiny logistic head. `embed_fn(waveform, sr) -> vector`."""

    embed_fn: object = None
    w: np.ndarray | None = None
    b: float = 0.0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _embed(self, clips: list[np.ndarray], sr: int) -> np.ndarray:
        fn = self.embed_fn or (lambda w, s: mel_stats_embedding(log_mel(w, s)))
        return np.stack([fn(c, sr) for c in clips])

    def fit(self, clips: list[np.ndarray], labels: np.ndarray, sr: int,
            *, lr: float = 0.3, epochs: int = 400) -> "AudioHead":
        X = self._embed(clips, sr)
        y = np.asarray(labels, float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-9
        Xn = (X - self.mean_) / self.std_
        self.w = np.zeros(X.shape[1]); self.b = 0.0
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Xn @ self.w + self.b)))
            self.w -= lr * (Xn.T @ (p - y) / len(y) + 1e-3 * self.w)
            self.b -= lr * float((p - y).mean())
        return self

    def predict_proba(self, clips: list[np.ndarray], sr: int) -> np.ndarray:
        Xn = (self._embed(clips, sr) - self.mean_) / self.std_
        return 1 / (1 + np.exp(-(Xn @ self.w + self.b)))


# --------------------------------------------------------------------------- #
# Synthetic impulse generator (unit-test material only)
# --------------------------------------------------------------------------- #
def synth_impulse_clip(sr: int, dur_s: float, impulse_t: float, *, kind: str = "click",
                       noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """A noise floor with one synthetic transient at impulse_t: 'click' (broadband, short)
    or 'ring' (decaying metallic tone) — stand-ins for swish vs rim hit in plumbing tests."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    w = rng.normal(0, noise, n)
    i = int(impulse_t * sr)
    if kind == "click":
        L = int(0.01 * sr)
        w[i:i + L] += rng.normal(0, 1.0, min(L, n - i))
    elif kind == "ring":
        L = int(0.12 * sr)
        t = np.arange(min(L, n - i)) / sr
        w[i:i + len(t)] += 0.8 * np.exp(-t / 0.03) * np.sin(2 * np.pi * 1800 * t)
    return w
