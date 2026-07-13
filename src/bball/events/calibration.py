"""Probability calibration (plan §5.4, §8, ablation A9).

FSM margins are scores, not probabilities. We fit temperature scaling (one parameter,
preserves ranking, minimizes NLL within its family) and Platt scaling (two parameters) and
report reliability diagrams + ECE + Brier. Leakage discipline (review R6): calibration is
fit on val-cal sessions only, never on the FSM-tuning or test sessions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


@dataclass
class TemperatureScaler:
    T: float = 1.0

    def fit(self, margins: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        margins = np.asarray(margins, float)
        labels = np.asarray(labels, float)

        def obj(logT):
            T = np.exp(logT)   # keep T > 0
            return _nll(_sigmoid(margins / T), labels)

        res = minimize_scalar(obj, bounds=(-3.0, 3.0), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def predict(self, margins: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(margins, float) / self.T)


@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0

    def fit(self, margins: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        margins = np.asarray(margins, float)
        labels = np.asarray(labels, float)

        def obj(p):
            return _nll(_sigmoid(p[0] * margins + p[1]), labels)

        res = minimize(obj, x0=[1.0, 0.0], method="Nelder-Mead")
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict(self, margins: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * np.asarray(margins, float) + self.b)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return (bin_centers, bin_accuracy, bin_confidence, bin_count) for a reliability
    diagram. Empty bins are NaN."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.full(n_bins, np.nan)
    conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        count[b] = int(m.sum())
        if count[b] > 0:
            acc[b] = labels[m].mean()
            conf[b] = probs[m].mean()
    return centers, acc, conf, count


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    _, acc, conf, count = reliability_curve(probs, labels, n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    mask = count > 0
    return float(np.sum(count[mask] / total * np.abs(acc[mask] - conf[mask])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    return float(np.mean((probs - labels) ** 2))
