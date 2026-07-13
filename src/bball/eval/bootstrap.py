"""Per-session bootstrap confidence intervals (plan §8): sessions are the exchangeable
unit, so we resample sessions (not shots) to get honest CIs on headline metrics."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    session_stats: list[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    agg: Callable[[np.ndarray], float] = np.mean,
) -> dict:
    """Bootstrap CI for a per-session statistic by resampling sessions with replacement."""
    vals = np.asarray(session_stats, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([agg(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)])
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return {"point": float(agg(vals)), "lo": lo, "hi": hi, "n": int(len(vals))}


def paired_session_delta(
    stats_a: list[float], stats_b: list[float], *, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Bootstrap the paired per-session difference (A - B). A CI excluding 0 is the bar for
    claiming an improvement (plan §8: no claim without non-overlapping CIs / a paired test)."""
    a = np.asarray(stats_a, float)
    b = np.asarray(stats_b, float)
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "significant": False}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"delta": float(d.mean()), "lo": lo, "hi": hi, "significant": (lo > 0 or hi < 0)}
